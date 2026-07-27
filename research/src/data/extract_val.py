import os
import sys
import io
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=== TRÍCH XUẤT VALIDATION TEXT (DAY 4) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    # Dùng CLIPModel tổng thể để tránh lỗi tải weight riêng biệt của HuggingFace
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    model.eval()
    
    val_json_path = "data/json/cap.dress.val.json"
    print(f"1. Đang nạp {val_json_path}...")
    with open(val_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    hidden_dict = {}
    embed_dict = {}
    
    print(f"2. Bắt đầu trích xuất {len(data)} truy vấn...")
    
    with torch.no_grad():
        for idx, item in enumerate(tqdm(data, desc="Extracting")):
            caption = " and ".join(item['captions'])
            inputs = processor(text=caption, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Tính hidden states từ text_model
            outputs = model.text_model(**inputs)
            last_hidden_state = outputs.last_hidden_state # [1, seq_len, 512]
            
            # Tìm vị trí EOS token (argmax của input_ids)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=device),
                inputs.input_ids.to(torch.int).argmax(dim=-1),
            ]
            
            # Áp dụng text_projection để ra được 512-d zero-shot embedding
            text_embeds = model.text_projection(pooled_output)
            
            # [seq_len, 512] - Dùng cho BaselineFusion
            hidden_dict[idx] = last_hidden_state.squeeze(0).cpu()
            
            # [512] - Dùng cho CLIP Zero-shot
            embed_dict[idx] = text_embeds.squeeze(0).cpu()
            
    hidden_path = "data/features/dress_val_text_hidden.pt"
    embed_path = "data/features/dress_val_text_embeds.pt"
    
    torch.save(hidden_dict, hidden_path)
    torch.save(embed_dict, embed_path)
    
    print(f"\n3. Hoàn tất!")
    print(f" - Đã lưu Hidden States tại: {hidden_path}")
    print(f" - Đã lưu Projected Embeds tại: {embed_path}")

if __name__ == "__main__":
    main()
