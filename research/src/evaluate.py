import os
import sys
import io
import json
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data.dataset import FashionIQDataset, custom_collate_fn
from models.model import BaselineFusion

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_rank(sim_matrix, target_score):
    return (sim_matrix >= target_score).sum().item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate BaselineFusion model")
    parser.add_argument("--ckpt", type=str, nargs='+', default=["baseline_infonce_1024_best.pth", "baseline_triplet_1024_best.pth"], help="One or more checkpoints to evaluate")
    parser.add_argument("--features_dir", type=str, default="data/features", help="Thư mục chứa feature .pt files (dùng data/features_fashionclip cho FashionCLIP)")
    args = parser.parse_args()

    print("=== ĐÁNH GIÁ MÔ HÌNH (DAY 5) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    print(f"[INFO] Features directory: {os.path.abspath(args.features_dir)}")
    
    print(f"\n[INFO] Checkpoints to evaluate: {args.ckpt}")
    for ckpt_file in args.ckpt:
        ckpt_path = os.path.abspath(f"checkpoints/{ckpt_file}")
        mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else 0
        print(f"  - {ckpt_file} (modified: {time.ctime(mtime)})")
    
    features_dir = args.features_dir
    
    print("\n1. Đang nạp Gallery (Kho ảnh 74,381)...")
    gallery_cls_768 = torch.load(os.path.join(features_dir, "gallery_cls_768.pt"), map_location=device)
    gallery_embeds_512 = torch.load(os.path.join(features_dir, "gallery_embeds_512.pt"), map_location=device)
    with open(os.path.join(features_dir, "gallery_asins.json"), "r") as f:
        gallery_asins = json.load(f)
        gallery_asin_to_idx = {asin: idx for idx, asin in enumerate(gallery_asins)}
        
    print("\n3. Đang nạp Validation Queries (Tất cả danh mục)...")
    val_data_all = []
    val_hidden_all = []
    val_embeds_all = []
    
    for cat in ['dress', 'shirt', 'toptee']:
        json_path = f"data/json/cap.{cat}.val.json"
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            val_data_all.extend(data)
            
        hidden_dict = torch.load(os.path.join(features_dir, f"{cat}_val_text_hidden.pt"), map_location=device)
        embed_dict = torch.load(os.path.join(features_dir, f"{cat}_val_text_embeds.pt"), map_location=device)
        
        # Flatten dict into list based on indices
        val_hidden_all.extend([hidden_dict[i] for i in range(len(hidden_dict))])
        val_embeds_all.extend([embed_dict[i] for i in range(len(embed_dict))])
    
    # Initialize counts
    recall_10_zs = recall_50_zs = 0
    
    # Load multiple models
    models = []
    model_names = []
    recalls_10 = []
    recalls_50 = []
    
    for ckpt_file in args.ckpt:
        print(f"   Khởi tạo BaselineFusion từ {ckpt_file}...")
        state_dict = torch.load(f"checkpoints/{ckpt_file}", map_location=device, weights_only=True)
        detected_hidden_dim = state_dict['mlp.0.weight'].shape[0] if 'mlp.0.weight' in state_dict else 1024
        
        model = BaselineFusion(hidden_dim=detected_hidden_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        
        # Derive run name from checkpoint filename
        run_name = ckpt_file.replace("_best.pth", "").replace("_last.pth", "").replace(".pth", "")
        model_names.append(run_name)
        recalls_10.append(0)
        recalls_50.append(0)
    
    print(f"\n5. BẮT ĐẦU ĐÁNH GIÁ (Cross-modal Retrieval) trên {len(val_data_all)} queries...")
    
    valid_queries = 0
    
    gallery_embeds_512_norm = F.normalize(gallery_embeds_512, p=2, dim=-1)
    gallery_cls_norm = F.normalize(gallery_cls_768, p=2, dim=-1)
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(val_data_all)):
            candidate_asin = item['candidate']
            target_asin = item['target']
            
            if candidate_asin not in gallery_asin_to_idx or target_asin not in gallery_asin_to_idx:
                continue
                
            candidate_idx = gallery_asin_to_idx[candidate_asin]
            target_idx = gallery_asin_to_idx[target_asin]
            valid_queries += 1
            
            # --- ĐỐI THỦ 1: CLIP ZERO-SHOT ---
            c_embed = F.normalize(gallery_embeds_512[candidate_idx].unsqueeze(0), p=2, dim=-1)
            t_embed = F.normalize(val_embeds_all[i].unsqueeze(0), p=2, dim=-1)
            zs_query = F.normalize(c_embed + t_embed, p=2, dim=-1)
            
            zs_sims = (zs_query @ gallery_embeds_512_norm.T).squeeze(0)
            zs_target_score = zs_sims[target_idx].item()
            zs_rank = get_rank(zs_sims, zs_target_score)
            if zs_rank <= 10: recall_10_zs += 1
            if zs_rank <= 50: recall_50_zs += 1
            
            # --- ĐỐI THỦ 2+: BASELINE FUSION (tất cả checkpoints) ---
            c_cls = gallery_cls_768[candidate_idx].unsqueeze(0) # [1, 768]
            t_hidden = val_hidden_all[i].unsqueeze(0) # [1, seq_len, 512]
            t_eos = t_hidden[:, -1, :] # [1, 512] - Vị trí -1 luôn là EOS vì dữ liệu được trích xuất từng câu đơn lẻ (không padding)
            
            for m_idx, model in enumerate(models):
                bf_query = model(c_cls, t_eos)
                bf_query_norm = F.normalize(bf_query, p=2, dim=-1)
                
                bf_sims = (bf_query_norm @ gallery_cls_norm.T).squeeze(0)
                bf_target_score = bf_sims[target_idx].item()
                bf_rank = get_rank(bf_sims, bf_target_score)
                if bf_rank <= 10: recalls_10[m_idx] += 1
                if bf_rank <= 50: recalls_50[m_idx] += 1
            
    print("\n=== KẾT QUẢ TỔNG HỢP ===")
    print(f"Tổng số truy vấn hợp lệ: {valid_queries}/{len(val_data_all)}")
    
    if valid_queries == 0:
        print("\n[CẢNH BÁO] Không có truy vấn hợp lệ nào! Kiểm tra lại --features_dir và gallery_asins.json")
        return
        
    print("\n1. CLIP ZERO-SHOT (Vector Addition)")
    print(f" - Recall@10: {recall_10_zs / valid_queries * 100:.2f}%")
    print(f" - Recall@50: {recall_50_zs / valid_queries * 100:.2f}%")
    
    for m_idx, name in enumerate(model_names):
        print(f"\n{m_idx + 2}. BASELINE FUSION ({name})")
        print(f" - Recall@10: {recalls_10[m_idx] / valid_queries * 100:.2f}%")
        print(f" - Recall@50: {recalls_50[m_idx] / valid_queries * 100:.2f}%")

if __name__ == "__main__":
    main()
