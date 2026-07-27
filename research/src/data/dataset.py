import os
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class FashionIQDataset(Dataset):
    def __init__(self, data_dir="data", category="dress"):
        """
        Khởi tạo Dataset. 
        Sẽ nạp toàn bộ đặc trưng ảnh (11GB) và đặc trưng chữ vào RAM.
        """
        self.data_dir = data_dir
        self.category = category
        
        # 1. Đọc file JSON và Text features
        self.data = []
        self.text_features = {}
        
        categories = ["dress", "shirt", "toptee"] if category == "all" else [category]
        
        global_idx = 0
        for cat in categories:
            # Load JSON
            json_path = os.path.join(data_dir, f"json/cap.{cat}.train.json")
            print(f"Loading metadata from {json_path}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                cat_data = json.load(f)
                self.data.extend(cat_data)
                
            # Load Text Features
            text_feat_path = os.path.join(data_dir, f"features/{cat}_text_tokens.pt")
            print(f"Loading text features from {text_feat_path}...")
            cat_text_features = torch.load(text_feat_path, map_location="cpu", weights_only=True)
            
            # Merge dictionary and shift indices
            for i in range(len(cat_data)):
                self.text_features[global_idx] = cat_text_features[i]
                global_idx += 1
                
        # 2. Đọc file đặc trưng ảnh (Chỉ lấy CLS - Cực nhẹ)
        image_feat_path = os.path.join(data_dir, "features/gallery_cls_768.pt")
        print(f"Loading image features (CLS only) from {image_feat_path}...")
        self.image_features = torch.load(image_feat_path, map_location="cpu", weights_only=True)
        print(f"Loaded {len(self.image_features)} image features.")
        
        # Load mapping file
        with open(os.path.join(data_dir, "features/gallery_asins.json"), "r") as f:
            self.gallery_asins = json.load(f)
            
        # Create an asin to index mapping for fast lookup
        self.asin_to_idx = {asin: idx for idx, asin in enumerate(self.gallery_asins)}
        
        print(f"Total pairs loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        candidate_id = item['candidate']
        target_id = item['target']
        
        # Lấy đặc trưng (shape: [768] cho ảnh, [seq_len, 768] cho text)
        src_idx = self.asin_to_idx[candidate_id]
        tgt_idx = self.asin_to_idx[target_id]
        
        src_img_feat = self.image_features[src_idx]
        tgt_img_feat = self.image_features[tgt_idx]
        txt_feat = self.text_features[idx] # Dictionary lưu theo số thứ tự
        
        return src_img_feat, txt_feat, tgt_img_feat

def custom_collate_fn(batch):
    """
    Xử lý Batch: Đệm (pad) các chuỗi Text có độ dài khác nhau về cùng 1 max_length.
    """
    src_imgs = []
    txt_feats = []
    tgt_imgs = []
    
    for src, txt, tgt in batch:
        src_imgs.append(src)
        txt_feats.append(txt)
        tgt_imgs.append(tgt)
        
    # Image features (CLS only) có shape cố định [768], ta chỉ cần stack lại
    src_imgs_batched = torch.stack(src_imgs) # [batch, 768]
    tgt_imgs_batched = torch.stack(tgt_imgs) # [batch, 768]
    
    # Text features có độ dài (seq_len) khác nhau -> Phải đệm (Pad)
    # pad_sequence mong đợi list các tensor có dạng [L, *] và trả về [L, batch, *]
    # Truyền batch_first=True để trả về [batch, L, *]
    txt_feats_padded = pad_sequence(txt_feats, batch_first=True, padding_value=0.0) # [batch, max_seq_len, 768]
    
    # Tạo text_attention_mask (True ở các vị trí có dữ liệu, False ở các vị trí padding)
    # Lấy chiều dài thật sự của từng text
    text_lengths = torch.tensor([t.size(0) for t in txt_feats], dtype=torch.long)
    max_len = txt_feats_padded.size(1)
    batch_size = len(batch)
    
    # Tạo mask: [batch, max_len]
    mask = torch.arange(max_len).expand(batch_size, max_len) < text_lengths.unsqueeze(1)
    
    return src_imgs_batched, txt_feats_padded, mask, text_lengths, tgt_imgs_batched
