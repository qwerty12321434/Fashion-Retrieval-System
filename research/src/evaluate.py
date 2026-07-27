import os
import sys
import io
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data.dataset import FashionIQDataset, custom_collate_fn
from models.model import BaselineFusion

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_rank(sim_matrix, target_score):
    return (sim_matrix >= target_score).sum().item()

def main():
    print("=== ĐÁNH GIÁ MÔ HÌNH (DAY 5) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    features_dir = "data/features"
    
    print("\n1. Đang nạp Gallery (Kho ảnh 74,381)...")
    gallery_cls_768 = torch.load(os.path.join(features_dir, "gallery_cls_768.pt"), map_location=device)
    gallery_embeds_512 = torch.load(os.path.join(features_dir, "gallery_embeds_512.pt"), map_location=device)
    with open(os.path.join(features_dir, "gallery_asins.json"), "r") as f:
        gallery_asins = json.load(f)
        
    print("\n3. Đang nạp Validation Queries...")
    with open("data/json/cap.dress.val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
        
    val_hidden = torch.load(os.path.join(features_dir, "dress_val_text_hidden.pt"), map_location=device)
    val_embeds = torch.load(os.path.join(features_dir, "dress_val_text_embeds.pt"), map_location=device)
    
    # Initialize counts
    recall_10_zs = recall_50_zs = 0
    recall_10_base = recall_50_base = 0
    
    print("   Khởi tạo BaselineFusion...")
    model_baseline = BaselineFusion(hidden_dim=512).to(device)
    model_baseline.load_state_dict(torch.load("checkpoints/baseline_all_best.pth", map_location=device, weights_only=True))
    model_baseline.eval()
    
    print(f"\n5. BẮT ĐẦU ĐÁNH GIÁ (Cross-modal Retrieval) trên {len(val_data)} queries...")
    
    valid_queries = 0
    
    gallery_embeds_512_norm = F.normalize(gallery_embeds_512, p=2, dim=-1)
    gallery_cls_norm = F.normalize(gallery_cls_768, p=2, dim=-1)
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(val_data)):
            candidate_asin = item['candidate']
            target_asin = item['target']
            
            if candidate_asin not in gallery_asins or target_asin not in gallery_asins:
                continue
                
            candidate_idx = gallery_asins.index(candidate_asin)
            target_idx = gallery_asins.index(target_asin)
            valid_queries += 1
            
            # --- ĐỐI THỦ 1: CLIP ZERO-SHOT ---
            c_embed = F.normalize(gallery_embeds_512[candidate_idx].unsqueeze(0), p=2, dim=-1)
            t_embed = F.normalize(val_embeds[i].unsqueeze(0), p=2, dim=-1)
            zs_query = F.normalize(c_embed + t_embed, p=2, dim=-1)
            
            zs_sims = (zs_query @ gallery_embeds_512_norm.T).squeeze(0)
            zs_target_score = zs_sims[target_idx].item()
            zs_rank = get_rank(zs_sims, zs_target_score)
            if zs_rank <= 10: recall_10_zs += 1
            if zs_rank <= 50: recall_50_zs += 1
            
            # --- ĐỐI THỦ 2: BASELINE FUSION ---
            c_cls = gallery_cls_768[candidate_idx].unsqueeze(0) # [1, 768]
            t_hidden = val_hidden[i].unsqueeze(0) # [1, seq_len, 512]
            t_eos = t_hidden[:, -1, :] # [1, 512]
            
            bf_query = model_baseline(c_cls, t_eos)
            bf_query_norm = F.normalize(bf_query, p=2, dim=-1)
            
            bf_sims = (bf_query_norm @ gallery_cls_norm.T).squeeze(0)
            bf_target_score = bf_sims[target_idx].item()
            bf_rank = get_rank(bf_sims, bf_target_score)
            if bf_rank <= 10: recall_10_base += 1
            if bf_rank <= 50: recall_50_base += 1
            
    print("\n=== KẾT QUẢ TỔNG HỢP ===")
    print(f"Tổng số truy vấn hợp lệ: {valid_queries}/{len(val_data)}")
    
    print("\n1. CLIP ZERO-SHOT (Vector Addition)")
    print(f" - Recall@10: {recall_10_zs / valid_queries * 100:.2f}%")
    print(f" - Recall@50: {recall_50_zs / valid_queries * 100:.2f}%")
    
    print("\n2. BASELINE FUSION (Train on ALL)")
    print(f" - Recall@10: {(recall_10_base / valid_queries * 100):.2f}%")
    print(f" - Recall@50: {(recall_50_base / valid_queries * 100):.2f}%")

if __name__ == "__main__":
    main()
