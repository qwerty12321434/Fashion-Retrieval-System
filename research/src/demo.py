import os
import sys
import io
import time
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import textwrap
from transformers import CLIPProcessor, CLIPModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from models.model import BaselineFusion

def main():
    parser = argparse.ArgumentParser(description="Demo Trực quan CIR với BaselineFusion vs Zero-shot")
    parser.add_argument("--candidate", type=str, required=True, help="ASIN của ảnh gốc")
    parser.add_argument("--text", type=str, required=True, help="Câu lệnh thay đổi (modifier)")
    parser.add_argument("--output", type=str, default="demo_result.png", help="Đường dẫn lưu ảnh kết quả")
    parser.add_argument("--ckpt", type=str, default="baseline_all_best.pth", help="Checkpoint to load")
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch32", help="Mô hình backbone sử dụng")
    parser.add_argument("--features_dir", type=str, default="data/features", help="Thư mục chứa feature của kho ảnh")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    ckpt_path = os.path.abspath(f"checkpoints/{args.ckpt}")
    mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else 0
    print(f"\n[INFO] Loading checkpoint from: {ckpt_path}")
    print(f"[INFO] Checkpoint modified time: {time.ctime(mtime)}\n")

    # 1. Khởi tạo mô hình
    print(f"1. Nạp CLIP ({args.backbone}) và BaselineFusion...")
    model_name = args.backbone
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    clip_model.eval()

    state_dict = torch.load(f"checkpoints/{args.ckpt}", map_location=device, weights_only=True)
    detected_hidden_dim = state_dict['mlp.0.weight'].shape[0] if 'mlp.0.weight' in state_dict else 1024
    fusion_model = BaselineFusion(hidden_dim=detected_hidden_dim).to(device)
    fusion_model.load_state_dict(state_dict)
    fusion_model.eval()

    # 2. Nạp Gallery
    print(f"2. Nạp kho ảnh Gallery từ {args.features_dir}...")
    features_dir = args.features_dir
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
    t_eos = txt_hidden[
        torch.arange(txt_hidden.shape[0], device=device),
        txt_inputs.input_ids.to(torch.int).argmax(dim=-1)
    ] # Lấy EOS token của text bằng vị trí có ID cao nhất (EOS)
    
    with torch.no_grad():
        bf_query = fusion_model(candidate_cls, t_eos) # [1, 768]
        
    bf_query_norm = F.normalize(bf_query, p=2, dim=-1)
    gallery_cls_norm = F.normalize(gallery_cls_768, p=2, dim=-1)
    bf_sims = (bf_query_norm @ gallery_cls_norm.T).squeeze(0)
    bf_scores, bf_top5_idx = torch.topk(bf_sims, k=5)
    bf_asins = [gallery_asins[idx] for idx in bf_top5_idx]
    # 7. Vẽ biểu đồ trực quan bằng Pillow (Thay cho matplotlib để tránh lỗi DLL Policy)
    print("7. Tạo ảnh trực quan...")
    
    CELL_W, CELL_H, TEXT_H, PADDING = 300, 400, 80, 20
    TOTAL_W = CELL_W * 6
    TOTAL_H = (CELL_H + TEXT_H) * 2
    
    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), "white")
    draw = ImageDraw.Draw(canvas)
    
    try:
        # Thử load font Arial trên Windows
        font = ImageFont.truetype("arial.ttf", 16)
        font_bold = ImageFont.truetype("arialbd.ttf", 18)
    except IOError:
        font = font_bold = ImageFont.load_default()
        
    def draw_cell(x, y, img_path, title_lines, title_color="black"):
        # Vẽ ảnh
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((CELL_W - 2*PADDING, CELL_H - 2*PADDING))
                img_x = x + (CELL_W - img.width) // 2
                img_y = y + TEXT_H + (CELL_H - img.height) // 2
                canvas.paste(img, (img_x, img_y))
            except Exception:
                draw.text((x + PADDING, y + TEXT_H + PADDING), "Image Load Error", fill="red", font=font_bold)
        else:
            draw.text((x + PADDING, y + TEXT_H + PADDING), f"Image Missing\n{os.path.basename(img_path)}", fill="red", font=font_bold)
            
        # Vẽ chữ
        text_y = y + 10
        for line in title_lines:
            f = font_bold if "ZERO-SHOT" in line or "FUSION" in line or "QUERY" in line else font
            draw.text((x + PADDING, text_y), line, fill=title_color, font=f)
            text_y += 22
            
    # Cột 0: Query
    q_x = 0
    q_y = (TOTAL_H - (CELL_H + TEXT_H)) // 2  # Căn giữa theo chiều dọc
    query_lines = ["QUERY", f"ASIN: {args.candidate}", "", "TEXT:"] + textwrap.wrap(f"'{args.text}'", width=25)
    draw_cell(q_x, q_y, candidate_path, query_lines, "blue")
    
    # Hàng 0: Zero-shot
    for i, (asin, score) in enumerate(zip(zs_asins, zs_scores)):
        img_path = os.path.join(image_dir, f"{asin}.jpg")
        if not os.path.exists(img_path): img_path = os.path.join(image_dir, f"{asin}.png")
        lines = ["[CLIP ZERO-SHOT]"] if i == 0 else []
        lines.extend([f"Top {i+1} ASIN: {asin}", f"Score: {score.item():.3f}"])
        draw_cell((i + 1) * CELL_W, 0, img_path, lines, "red" if i == 0 else "black")
        
    # Hàng 1: BaselineFusion
    for i, (asin, score) in enumerate(zip(bf_asins, bf_scores)):
        img_path = os.path.join(image_dir, f"{asin}.jpg")
        if not os.path.exists(img_path): img_path = os.path.join(image_dir, f"{asin}.png")
        lines = ["[BASELINE FUSION]"] if i == 0 else []
        lines.extend([f"Top {i+1} ASIN: {asin}", f"Score: {score.item():.3f}"])
        draw_cell((i + 1) * CELL_W, CELL_H + TEXT_H, img_path, lines, "green" if i == 0 else "black")
        
    canvas.save(args.output)
    print(f"\n=> Đã lưu kết quả tại: {args.output}")

if __name__ == "__main__":
    main()
