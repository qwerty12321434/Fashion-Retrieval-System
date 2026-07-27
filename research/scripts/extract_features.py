import os
import sys
import io
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
import glob

# Ép Windows Terminal dùng UTF-8 để không bị lỗi font tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Cấu hình đường dẫn
IMAGE_DIR = r"E:\MyDownloads\fashion-iq-dataset\fashionIQ_dataset\images"
OUTPUT_DIR = "data/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tối ưu hóa: Dùng Dataset và DataLoader để chạy Batch
def custom_collate(batch):
    return tuple(zip(*batch))

class FashionImageDataset(Dataset):
    def __init__(self, image_dir, image_files, processor):
        self.image_dir = image_dir
        self.image_files = image_files
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        asin = img_file.split('.')[0]
        img_path = os.path.join(self.image_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            # Chỉ preprocess ảnh trả về tensor
            inputs = self.processor(images=image, return_tensors="pt")
            return asin, inputs['pixel_values'].squeeze(0)
        except Exception as e:
            return asin, None

def extract_image_features(model, processor, device):
    print("--- Trích xuất Image Features ---")
    if not os.path.exists(IMAGE_DIR):
        print(f"Thư mục {IMAGE_DIR} không tồn tại.")
        return
        
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    if len(image_files) == 0:
        print("Không tìm thấy ảnh.")
        return
        
    dataset = FashionImageDataset(IMAGE_DIR, image_files, processor)
    
    # RTX 4050 6GB VRAM có thể gánh batch_size 128 hoặc 256. Để 128 là cực kỳ an toàn.
    # num_workers=4 giúp CPU đọc ảnh song song nhanh hơn rất nhiều.
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=custom_collate)
    
    image_features_dict = {}
    
    with torch.no_grad():
        for batch_asins, batch_images in tqdm(dataloader):
            valid_idx = [i for i, img in enumerate(batch_images) if img is not None]
            if not valid_idx: continue
            
            valid_asins = [batch_asins[i] for i in valid_idx]
            valid_images = torch.stack([batch_images[i] for i in valid_idx]).to(device)
            
            outputs = model.vision_model(pixel_values=valid_images)
            tokens_batch = outputs.last_hidden_state.cpu()
            
            for i, asin in enumerate(valid_asins):
                image_features_dict[asin] = tokens_batch[i]
                
    torch.save(image_features_dict, os.path.join(OUTPUT_DIR, "all_image_tokens.pt"))
    print(f"Đã lưu {len(image_features_dict)} image features vào all_image_tokens.pt!")

def extract_text_features(model, processor, device):
    print("--- Trích xuất Text Features ---")
    json_files = glob.glob("data/json/*.json")
    
    if not json_files:
        print("Không tìm thấy file JSON nào trong data/json/")
        return
        
    for json_path in json_files:
        print(f"\nĐang xử lý: {os.path.basename(json_path)}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        text_features_dict = {}
        with torch.no_grad():
            for idx, item in enumerate(tqdm(data)):
                text = " and ".join(item['captions'])
                inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model.text_model(**inputs)
                text_features_dict[idx] = outputs.last_hidden_state.squeeze(0).cpu()
                
        category = os.path.basename(json_path).split('.')[1]
        output_filename = f"{category}_text_tokens.pt"
        torch.save(text_features_dict, os.path.join(OUTPUT_DIR, output_filename))
        print(f"Đã lưu {len(text_features_dict)} text features vào {output_filename}!")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Đưa việc khởi tạo Model vào trong main() để tránh lỗi kẹt GPU khi dùng num_workers > 0 trên Windows
    model_name = "openai/clip-vit-base-patch32" # model Vision Transformer (ViT) của OpenAI
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device) 
    model.eval()

    # Bật lại cả 2 hàm để extract toàn bộ từ đầu
    extract_image_features(model, processor, device)
    extract_text_features(model, processor, device)
    print("🎉 Hoàn thành xuất đặc trưng Ngày 1!")

if __name__ == "__main__":
    main()
