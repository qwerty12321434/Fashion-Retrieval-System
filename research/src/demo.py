import os
import sys
import io
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from transformers import CLIPProcessor, CLIPModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from models.model import BaselineFusion

def main():
    parser = argparse.ArgumentParser(description="Demo Trực quan CIR với BaselineFusion vs Zero-shot")
    parser.add_argument("--candidate", type=str, required=True, help="ASIN của ảnh gốc")
    parser.add_argument("--text", type=str, required=True, help="Câu lệnh thay đổi (modifier)")
    parser.add_argument("--output", type=str, default="demo_result.png", help="Đường dẫn lưu ảnh kết quả")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo mô hình
    print("1. Nạp CLIP và BaselineFusion...")
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    clip_model.eval()

    fusion_model = BaselineFusion(hidden_dim=512).to(device)
    fusion_model.load_state_dict(torch.load("checkpoints/baseline_all_best.pth", map_location=device, weights_only=True))
    fusion_model.eval()

    # 2. Nạp Gallery
    print("2. Nạp kho ảnh Gallery...")
    features_dir = "data/features"
    gallery_cls_768 = torch.load(os.path.join(features_dir, "gallery_cls_768.pt"), map_location=device)
    gallery_embeds_512 = torch.load(os.path.join(features_dir, "gallery_embeds_512.pt"), map_location=device)
    with open(os.path.join(features_dir, "gallery_asins.json"), "r") as f:
        gallery_asins = json.load(f)

    # 3. Trích xuất đặc trưng cho Candidate Image
    image_dir = r"E:\MyDownloads\fashion-iq-dataset\fashionIQ_dataset\images"
    candidate_path = os.path.join(image_dir, f"{args.candidate}.jpg")
    if not os.path.exists(candidate_path):
        candidate_path = os.path.join(image_dir, f"{args.candidate}.png")
        
    print(f"3. Xử lý Candidate: {candidate_path}")
    image = Image.open(candidate_path).convert("RGB")
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_outputs = clip_model.vision_model(**img_inputs)
        candidate_50_tokens = img_outputs.last_hidden_state # [1, 50, 768]
        candidate_cls = candidate_50_tokens[:, 0, :] # [1, 768] (BaselineFusion)
        candidate_embed = clip_model.visual_projection(clip_model.vision_model.post_layernorm(candidate_cls)) # [1, 512] (Zero-shot)

    # 4. Trích xuất đặc trưng cho Text
    print(f"4. Xử lý Text: '{args.text}'")
    txt_inputs = processor(text=args.text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        txt_outputs = clip_model.text_model(**txt_inputs)
        txt_hidden = txt_outputs.last_hidden_state # [1, seq_len, 512] (BaselineFusion)
        
        pooled_output = txt_hidden[
            torch.arange(txt_hidden.shape[0], device=device),
            txt_inputs.input_ids.to(torch.int).argmax(dim=-1),
        ]
        text_embed = clip_model.text_projection(pooled_output) # [1, 512] (Zero-shot)
        
    # 5. Xử lý Zero-shot CLIP
    print("5. Xử lý Zero-shot CLIP...")
    c_embed_norm = F.normalize(candidate_embed, p=2, dim=-1)
    t_embed_norm = F.normalize(text_embed, p=2, dim=-1)
    zs_query = F.normalize(c_embed_norm + t_embed_norm, p=2, dim=-1)
    gallery_embeds_norm = F.normalize(gallery_embeds_512, p=2, dim=-1)
    zs_sims = (zs_query @ gallery_embeds_norm.T).squeeze(0)
    zs_scores, zs_top5_idx = torch.topk(zs_sims, k=5)
    zs_asins = [gallery_asins[idx] for idx in zs_top5_idx]

    # 6. Xử lý BaselineFusion
    print("6. Xử lý BaselineFusion...")
    t_eos = txt_hidden[:, -1, :] # Lấy EOS token của text
    
    with torch.no_grad():
        bf_query = fusion_model(candidate_cls, t_eos) # [1, 768]
        
    bf_query_norm = F.normalize(bf_query, p=2, dim=-1)
    gallery_cls_norm = F.normalize(gallery_cls_768, p=2, dim=-1)
    bf_sims = (bf_query_norm @ gallery_cls_norm.T).squeeze(0)
    bf_scores, bf_top5_idx = torch.topk(bf_sims, k=5)
    bf_asins = [gallery_asins[idx] for idx in bf_top5_idx]
    
    # 7. Vẽ biểu đồ trực quan (2 Rows)
    print("7. Tạo ảnh trực quan...")
    fig = plt.figure(figsize=(24, 10))
    gs = GridSpec(2, 6, figure=fig)
    
    # Cột bên trái: Ảnh Candidate + Text
    ax_query = fig.add_subplot(gs[0:2, 0])
    ax_query.imshow(image)
    ax_query.set_title(f"QUERY\nASIN: {args.candidate}\n\n+ TEXT:\n'{args.text}'", fontsize=14, loc='center', color='blue')
    ax_query.axis("off")
    
    # Hàng 1: ZERO-SHOT CLIP
    for i, asin in enumerate(zs_asins):
        ax = fig.add_subplot(gs[0, i+1])
        res_path = os.path.join(image_dir, f"{asin}.jpg")
        if not os.path.exists(res_path):
            res_path = os.path.join(image_dir, f"{asin}.png")
        try:
            res_img = Image.open(res_path).convert("RGB")
            ax.imshow(res_img)
            score = zs_scores[i].item()
            prefix = "[CLIP ZERO-SHOT]\n" if i == 0 else ""
            title_color = "red" if i == 0 else "black"
            ax.set_title(f"{prefix}Top {i+1} ASIN: {asin}\nScore: {score:.3f}", color=title_color, fontsize=12)
            ax.axis("off")
        except Exception:
            ax.set_title(f"Image Missing\n{asin}")
            ax.axis("off")

    # Hàng 2: BASELINE FUSION
    for i, asin in enumerate(bf_asins):
        ax = fig.add_subplot(gs[1, i+1])
        res_path = os.path.join(image_dir, f"{asin}.jpg")
        if not os.path.exists(res_path):
            res_path = os.path.join(image_dir, f"{asin}.png")
        try:
            res_img = Image.open(res_path).convert("RGB")
            ax.imshow(res_img)
            score = bf_scores[i].item()
            prefix = "[BASELINE FUSION]\n" if i == 0 else ""
            title_color = "green" if i == 0 else "black"
            ax.set_title(f"{prefix}Top {i+1} ASIN: {asin}\nScore: {score:.3f}", color=title_color, fontsize=12)
            ax.axis("off")
        except Exception:
            ax.set_title(f"Image Missing\n{asin}")
            ax.axis("off")
            
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"\n=> Đã lưu kết quả tại: {args.output}")

if __name__ == "__main__":
    main()
