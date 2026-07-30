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

# Ép Windows Terminal dùng UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================================
# CẤU HÌNH
# =============================================================================
MODEL_NAME = "patrickjohncyh/fashion-clip"  # FashionCLIP — fine-tuned trên ~700K sản phẩm thời trang
IMAGE_DIR  = r"E:\MyDownloads\fashion-iq-dataset\fashionIQ_dataset\images"
OUTPUT_DIR = "data/features_fashionclip"    # Thư mục RIÊNG, không đè lên bản CLIP-base cũ
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATASET CHO ẢNH (Batch loading)
# =============================================================================
def custom_collate(batch):
    return tuple(zip(*batch))

class FashionImageDataset(Dataset):
    def __init__(self, image_dir, image_files, processor):
        self.image_dir   = image_dir
        self.image_files = image_files
        self.processor   = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        asin     = img_file.split('.')[0]
        img_path = os.path.join(self.image_dir, img_file)
        try:
            image  = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return asin, inputs['pixel_values'].squeeze(0)
        except Exception:
            return asin, None

# =============================================================================
# BƯỚC 1: TRÍCH XUẤT TOÀN BỘ IMAGE TOKENS (all_image_tokens.pt)
# =============================================================================
def extract_image_features(model, processor, device):
    """
    Trích xuất last_hidden_state của vision_model cho toàn bộ ảnh trong IMAGE_DIR.
    Output: all_image_tokens.pt  — dict{asin: tensor[50, 768]}
    Dung lượng ước tính: ~11GB (giống CLIP-base vì cùng ViT-B/32 architecture)
    """
    print("\n" + "="*60)
    print("[BƯỚC 1/4] Trích xuất Image Features (FashionCLIP Vision)")
    print("="*60)

    if not os.path.exists(IMAGE_DIR):
        print(f"  [LỖI] Thư mục ảnh không tồn tại: {IMAGE_DIR}")
        return

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print("  [LỖI] Không tìm thấy ảnh trong IMAGE_DIR.")
        return

    print(f"  Tổng số ảnh: {len(image_files)}")
    print(f"  Model: {MODEL_NAME}")

    dataset    = FashionImageDataset(IMAGE_DIR, image_files, processor)
    # num_workers=4 tăng tốc đọc ảnh trên Windows (đã an toàn vì model init trong main + if __name__ == "__main__" guard)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False,
                            num_workers=4, collate_fn=custom_collate)

    image_features_dict = {}

    with torch.no_grad():
        for batch_asins, batch_images in tqdm(dataloader, desc="Extracting images"):
            valid_idx    = [i for i, img in enumerate(batch_images) if img is not None]
            if not valid_idx:
                continue
            valid_asins  = [batch_asins[i] for i in valid_idx]
            valid_images = torch.stack([batch_images[i] for i in valid_idx]).to(device)

            outputs      = model.vision_model(pixel_values=valid_images)
            tokens_batch = outputs.last_hidden_state.cpu()  # [B, 50, 768]

            for i, asin in enumerate(valid_asins):
                image_features_dict[asin] = tokens_batch[i]

    out_path = os.path.join(OUTPUT_DIR, "all_image_tokens.pt")
    torch.save(image_features_dict, out_path)
    print(f"  [OK] Đã lưu {len(image_features_dict)} image tokens -> {out_path}")

# =============================================================================
# BƯỚC 2: CHUẨN BỊ GALLERY (gallery_cls_768.pt + gallery_embeds_512.pt)
# =============================================================================
def prepare_gallery(model, device):
    """
    Từ all_image_tokens.pt:
      - Cắt CLS token [0, :] -> gallery_cls_768.pt  [N, 768]
      - Chiếu qua post_layernorm + visual_projection -> gallery_embeds_512.pt  [N, 512]
      - Lưu danh sách ASIN -> gallery_asins.json
    """
    print("\n" + "="*60)
    print("[BƯỚC 2/4] Chuẩn bị Gallery (CLS + Projected Embeds)")
    print("="*60)

    all_tokens_path = os.path.join(OUTPUT_DIR, "all_image_tokens.pt")
    print(f"  Nạp {all_tokens_path} ...")
    all_image_tokens = torch.load(all_tokens_path)

    asins    = list(all_image_tokens.keys())
    cls_list = [all_image_tokens[asin][0, :] for asin in asins]
    gallery_cls_768 = torch.stack(cls_list)  # [N, 768]

    # Lưu danh sách ASIN
    asins_path = os.path.join(OUTPUT_DIR, "gallery_asins.json")
    with open(asins_path, "w") as f:
        json.dump(asins, f)
    print(f"  [OK] Đã lưu {len(asins)} ASINs -> {asins_path}")

    # Lưu CLS features
    cls_path = os.path.join(OUTPUT_DIR, "gallery_cls_768.pt")
    torch.save(gallery_cls_768, cls_path)
    print(f"  [OK] Đã lưu gallery_cls_768.pt  shape={gallery_cls_768.shape}")

    # Giải phóng RAM trước khi tính embeddings
    del all_image_tokens

    # Tính projected embeddings (512-dim) cho Zero-shot CLIP
    print("  Đang chiếu CLS qua post_layernorm + visual_projection...")
    batch_size         = 4096
    gallery_embeds_512 = []

    with torch.no_grad():
        for i in range(0, len(gallery_cls_768), batch_size):
            batch = gallery_cls_768[i:i+batch_size].to(device)
            x     = model.vision_model.post_layernorm(batch)
            x     = model.visual_projection(x)
            gallery_embeds_512.append(x.cpu())

    gallery_embeds_512 = torch.cat(gallery_embeds_512, dim=0)
    embeds_path        = os.path.join(OUTPUT_DIR, "gallery_embeds_512.pt")
    torch.save(gallery_embeds_512, embeds_path)
    print(f"  [OK] Đã lưu gallery_embeds_512.pt  shape={gallery_embeds_512.shape}")

# =============================================================================
# BƯỚC 3: TRÍCH XUẤT TEXT FEATURES (TRAIN)
# =============================================================================
def extract_train_text(model, processor, device):
    """
    Trích xuất last_hidden_state của text_model cho toàn bộ triplets train.
    Output: {dress,shirt,toptee}_text_tokens.pt  — dict{idx: tensor[seq_len, 512]}
    """
    print("\n" + "="*60)
    print("[BƯỚC 3/4] Trích xuất Train Text Features")
    print("="*60)

    json_files = glob.glob("data/json/cap.*.train.json")
    if not json_files:
        print("  [CANH BAO] Không tìm thấy file JSON train trong data/json/")
        return

    for json_path in json_files:
        print(f"\n  Đang xử lý: {os.path.basename(json_path)}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text_features_dict = {}
        with torch.no_grad():
            for idx, item in enumerate(tqdm(data, desc="  Train text")):
                text   = " and ".join(item['captions'])
                inputs = processor(text=text, return_tensors="pt",
                                   padding=True, truncation=True).to(device)
                outputs = model.text_model(**inputs)
                # last_hidden_state: [1, seq_len, 512]
                text_features_dict[idx] = outputs.last_hidden_state.squeeze(0).cpu()

        # Tên file dựa theo danh mục: cap.dress.train.json -> dress_text_tokens.pt
        category        = os.path.basename(json_path).split('.')[1]
        output_filename = f"{category}_text_tokens.pt"
        out_path        = os.path.join(OUTPUT_DIR, output_filename)
        torch.save(text_features_dict, out_path)
        print(f"  [OK] Đã lưu {len(text_features_dict)} train captions -> {out_path}")

# =============================================================================
# BƯỚC 4: TRÍCH XUẤT TEXT FEATURES (VALIDATION)
# =============================================================================
def extract_val_text(model, processor, device):
    """
    Trích xuất text features cho validation queries.
    Output (mỗi danh mục):
      - {cat}_val_text_hidden.pt  — dict{idx: tensor[seq_len, 512]}  (cho BaselineFusion)
      - {cat}_val_text_embeds.pt  — dict{idx: tensor[512]}           (cho Zero-shot CLIP)

    EOS token: dùng argmax(input_ids) để tìm đúng vị trí <|endoftext|> (ID=49407),
    nhất quán với extract_val.py và demo.py.
    """
    print("\n" + "="*60)
    print("[BƯỚC 4/4] Trích xuất Validation Text Features")
    print("="*60)

    categories = ['dress', 'shirt', 'toptee']

    for category in categories:
        val_json_path = f"data/json/cap.{category}.val.json"
        if not os.path.exists(val_json_path):
            print(f"  Bỏ qua {category}: không tìm thấy {val_json_path}")
            continue

        print(f"\n  Đang xử lý val/{category} ...")
        with open(val_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        hidden_dict = {}
        embed_dict  = {}

        with torch.no_grad():
            for idx, item in enumerate(tqdm(data, desc=f"  Val {category}")):
                caption = " and ".join(item['captions'])
                inputs  = processor(text=caption, return_tensors="pt",
                                    padding=True, truncation=True).to(device)

                outputs           = model.text_model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # [1, seq_len, 512]

                # EOS token: vị trí có input_id cao nhất (token 49407 = <|endoftext|>)
                eos_idx     = inputs.input_ids.to(torch.int).argmax(dim=-1)
                pooled      = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=device), eos_idx
                ]
                text_embeds = model.text_projection(pooled)  # [1, 512]

                hidden_dict[idx] = last_hidden_state.squeeze(0).cpu()  # [seq_len, 512]
                embed_dict[idx]  = text_embeds.squeeze(0).cpu()        # [512]

        hidden_path = os.path.join(OUTPUT_DIR, f"{category}_val_text_hidden.pt")
        embed_path  = os.path.join(OUTPUT_DIR, f"{category}_val_text_embeds.pt")
        torch.save(hidden_dict, hidden_path)
        torch.save(embed_dict,  embed_path)
        print(f"  [OK] {category}: {len(hidden_dict)} queries -> hidden + embeds đã lưu")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  TRÍCH XUẤT ĐẶC TRƯNG VỚI FASHIONCLIP")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print(f"\n  Đang tải FashionCLIP từ HuggingFace...")
    print(f"  (Lần đầu sẽ download ~600MB, lần sau dùng cache)")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model     = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
    model.eval()
    print(f"  [OK] FashionCLIP loaded!")

    # Xác nhận output dims đúng như kỳ vọng
    print(f"\n  [INFO] Vision hidden dim : {model.config.vision_config.hidden_size}  (kỳ vọng: 768)")
    print(f"  [INFO] Text hidden dim   : {model.config.text_config.hidden_size}   (kỳ vọng: 512)")
    print(f"  [INFO] Projection dim    : {model.config.projection_dim}            (kỳ vọng: 512)")

    extract_image_features(model, processor, device)  # ~30-60 phut
    prepare_gallery(model, device)                     # ~2-5 phut
    extract_train_text(model, processor, device)       # ~5-10 phut
    extract_val_text(model, processor, device)         # ~3-5 phut

    print("\n" + "=" * 60)
    print("  HOÀN TẤT! Toàn bộ FashionCLIP features đã lưu tại:")
    print(f"  {os.path.abspath(OUTPUT_DIR)}")
    print("\n  Để đánh giá với FashionCLIP features, chạy lệnh:")
    print("  python src/evaluate.py --features_dir data/features_fashionclip")
    print("\n  Để so sánh với CLIP-base (baseline), chạy:")
    print("  python src/evaluate.py --features_dir data/features")
    print("=" * 60)

if __name__ == "__main__":
    main()
