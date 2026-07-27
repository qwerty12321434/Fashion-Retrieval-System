import torch
from transformers import CLIPModel
import os
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=== CHUẨN BỊ GALLERY (DAY 4) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    features_dir = "data/features"
    all_tokens_path = os.path.join(features_dir, "all_image_tokens.pt")
    
    print(f"1. Đang nạp dictionary khổng lồ từ {all_tokens_path}...")
    all_image_tokens = torch.load(all_tokens_path)
    
    print("2. Đang tạo Tensor Gallery và danh sách ASIN...")
    asins = list(all_image_tokens.keys())
    
    # Gom tất cả các CLS token [768] thành tensor [N, 768]
    cls_list = []
    for asin in asins:
        cls_list.append(all_image_tokens[asin][0, :])
    gallery_cls_768 = torch.stack(cls_list)
    
    with open("data/features/gallery_asins.json", "w") as f:
        json.dump(asins, f)
        
    out_cls_path = os.path.join(features_dir, "gallery_cls_768.pt")
    torch.save(gallery_cls_768, out_cls_path)
    print(f"   Đã lưu: {out_cls_path} (Kích thước: {gallery_cls_768.shape})")
    print(f"   Đã lưu: data/features/gallery_asins.json")
    
    del all_image_tokens
    
    print("\n3. Khởi tạo mô hình CLIP để tính Projected Embeddings...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    model.eval()
    
    batch_size = 4096
    gallery_embeds_512 = []
    
    print("4. Đang chạy CLS qua post_layernorm và visual_projection...")
    with torch.no_grad():
        for i in range(0, len(gallery_cls_768), batch_size):
            batch = gallery_cls_768[i:i+batch_size].to(device)
            x = model.vision_model.post_layernorm(batch)
            x = model.visual_projection(x)
            gallery_embeds_512.append(x.cpu())
            
    gallery_embeds_512 = torch.cat(gallery_embeds_512, dim=0)
    
    out_embed_path = os.path.join(features_dir, "gallery_embeds_512.pt")
    torch.save(gallery_embeds_512, out_embed_path)
    print(f"   Đã lưu: {out_embed_path} (Kích thước: {gallery_embeds_512.shape})")
    print("\nHOÀN TẤT CHUẨN BỊ GALLERY!")

if __name__ == "__main__":
    main()
