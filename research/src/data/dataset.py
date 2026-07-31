import os
import json
import random
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence


def validate_patch_features(patch_features, required_candidates, expected_shape, context):
    """Dừng sớm nếu artifact visual tokens thiếu candidate hoặc sai shape."""
    required = set(required_candidates)
    missing = sorted(required - set(patch_features))
    if missing:
        raise RuntimeError(
            f"[AACL] Thiếu patch token cho {len(missing)} {context} candidate. "
            f"Ví dụ: {missing[:5]}. Hãy chạy lại scripts/prep_candidate_patches.py."
        )

    invalid = [
        (asin, tuple(patch_features[asin].shape))
        for asin in required
        if tuple(patch_features[asin].shape) != expected_shape
    ]
    if invalid:
        raise RuntimeError(
            f"[AACL] Có {len(invalid)} {context} patch tensor sai shape; "
            f"yêu cầu {expected_shape}. Ví dụ: {invalid[:5]}"
        )
    print(f"  Patch coverage OK: {len(required)} candidates, shape={expected_shape}.")


class FashionIQDataset(Dataset):
    def __init__(self, data_dir="data", features_dir="data/features", category="dress", use_patches=False):
        """
        Khởi tạo Dataset. 
        Sẽ nạp toàn bộ đặc trưng ảnh (228MB cho CLS token) và đặc trưng chữ vào RAM để tăng tốc.
        Args:
            use_patches: Nếu True, load thêm candidate_patch_tokens.pt
                         ([50, 768] visual tokens = 1 CLS + 49 spatial patches).
                         Cần chạy scripts/prep_candidate_patches.py trước.
        """
        self.data_dir     = data_dir
        self.features_dir = features_dir
        self.category     = category
        self.use_patches  = use_patches
        
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
                for item in cat_data:
                    item['category'] = cat
                self.data.extend(cat_data)
                
            # Load Text Features
            text_feat_path = os.path.join(features_dir, f"{cat}_text_tokens.pt")
            print(f"Loading text features from {text_feat_path}...")
            cat_text_features = torch.load(text_feat_path, map_location="cpu", weights_only=True)
            
            # Merge dictionary and shift indices
            for i in range(len(cat_data)):
                self.text_features[global_idx] = cat_text_features[i]
                global_idx += 1
                
        # 2. Đọc file đặc trưng ảnh (Chỉ lấy CLS - Cực nhẹ)
        image_feat_path = os.path.join(features_dir, "gallery_cls_768.pt")
        print(f"Loading image features (CLS only) from {image_feat_path}...")
        self.image_features = torch.load(image_feat_path, map_location="cpu", weights_only=True)
        print(f"Loaded {len(self.image_features)} image features.")
        
        # Load mapping file
        with open(os.path.join(features_dir, "gallery_asins.json"), "r") as f:
            self.gallery_asins = json.load(f)
            
        # Create an asin to index mapping for fast lookup
        self.asin_to_idx = {asin: idx for idx, asin in enumerate(self.gallery_asins)}
        
        print(f"Total pairs loaded: {len(self.data)}")
        
        # 3. (Optional) Load patch tokens cho candidate images (AACL mode)
        self.patch_features = None
        if use_patches:
            patch_path = os.path.join(features_dir, "candidate_patch_tokens.pt")
            if not os.path.exists(patch_path):
                raise FileNotFoundError(
                    f"[AACL] Không tìm thấy {patch_path}.\n"
                    "Hãy chạy trước: python scripts/prep_candidate_patches.py"
                )
            print(f"Loading candidate patch tokens (AACL) from {patch_path}...")
            # mmap tránh nạp toàn bộ artifact patch vào RAM ngay khi khởi tạo.
            # Điều này cũng giúp đọc được artifact cũ vốn giữ storage 11GB.
            self.patch_features = torch.load(
                patch_path, map_location="cpu", weights_only=True, mmap=True
            )
            print(f"  Loaded patch tokens for {len(self.patch_features)} candidate ASINs.")

            validate_patch_features(
                self.patch_features,
                (item["candidate"] for item in self.data),
                expected_shape=(50, self.image_features.shape[-1]),
                context="train",
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item         = self.data[idx]
        candidate_id = item['candidate']
        target_id    = item['target']
        
        # Lấy đặc trưng (shape: [768] cho ảnh, [seq_len, 512] cho text)
        src_idx = self.asin_to_idx[candidate_id]
        tgt_idx = self.asin_to_idx[target_id]
        
        src_img_feat = self.image_features[src_idx]   # [768] CLS
        tgt_img_feat = self.image_features[tgt_idx]   # [768] CLS
        txt_feat     = self.text_features[idx]         # [seq_len, 512] full sequence
        
        if self.use_patches:
            # [50, 768] — 1 CLS + 49 spatial patch tokens của candidate image.
            # Coverage đã được kiểm tra fail-fast trong __init__.
            src_patch = self.patch_features[candidate_id]
            return src_img_feat, src_patch, txt_feat, tgt_img_feat
        else:
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
    txt_feats_padded = pad_sequence(txt_feats, batch_first=True, padding_value=0.0) # [batch, max_seq_len, 512]
    
    # Tạo text_attention_mask (True ở các vị trí có dữ liệu, False ở các vị trí padding)
    # Lấy chiều dài thật sự của từng text
    text_lengths = torch.tensor([t.size(0) for t in txt_feats], dtype=torch.long)
    max_len = txt_feats_padded.size(1)
    batch_size = len(batch)
    
    # Tạo mask: [batch, max_len]
    mask = torch.arange(max_len).expand(batch_size, max_len) < text_lengths.unsqueeze(1)
    
    return src_imgs_batched, txt_feats_padded, mask, text_lengths, tgt_imgs_batched

class CategoryBatchSampler(Sampler):
    """
    Sampler đặc biệt: Đảm bảo mỗi Batch bốc ra chỉ chứa các mẫu thuộc CÙNG MỘT danh mục (Dress / Shirt / TopTee).
    Giúp InfoNCE loss không bị rơi vào bẫy 'Easy Negatives' (So sánh váy với áo sơ mi).
    """
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Nhóm các index theo category
        self.cat_indices = {}
        
        # Hỗ trợ cả Dataset gốc lẫn Subset
        is_subset = hasattr(dataset, 'dataset') and hasattr(dataset, 'indices')
        base_dataset = dataset.dataset if is_subset else dataset
        
        # Nếu là Subset, ta duyệt qua các indices được cấp
        iterable_indices = range(len(dataset))
        for idx in iterable_indices:
            # Lấy index thực tế trong base_dataset
            real_idx = dataset.indices[idx] if is_subset else idx
            item = base_dataset.data[real_idx]
            cat = item.get('category', 'unknown')
            if cat not in self.cat_indices:
                self.cat_indices[cat] = []
            # Ta lưu index tương đối của Subset (từ 0 đến len(Subset)-1) để DataLoader dùng
            self.cat_indices[cat].append(idx)
            
    def __iter__(self):
        # Tạo danh sách các batch
        batches = []
        for cat, indices in self.cat_indices.items():
            # Xáo trộn index trong nội bộ danh mục
            random.shuffle(indices)
            # Cắt thành các batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
                    
        # Xáo trộn thứ tự các batch (để mô hình không học 1 vệt toàn váy rồi mới đến áo)
        random.shuffle(batches)
        
        for batch in batches:
            yield batch
            
    def __len__(self):
        length = 0
        for cat, indices in self.cat_indices.items():
            if self.drop_last:
                length += len(indices) // self.batch_size
            else:
                length += (len(indices) + self.batch_size - 1) // self.batch_size
        return length


def aacl_collate_fn(batch):
    """
    Collate function cho AACL mode (use_patches=True).
    Mỗi sample trả về: (src_img_feat, src_patch, txt_feat, tgt_img_feat)
    
    Returns:
        src_imgs_batched : [B, 768]           — CLS token candidate (giữ để eval baseline song song)
        src_patches      : [B, 50, 768]       — 50 visual tokens candidate
        txt_feats_padded : [B, max_L, 512]    — full text sequence (đã pad)
        txt_mask         : [B, max_L] bool    — True = real token
        txt_lengths      : [B]                — độ dài thật sự
        tgt_imgs_batched : [B, 768]           — CLS token target
    """
    src_imgs  = []
    src_pats  = []
    txt_feats = []
    tgt_imgs  = []

    for src, patch, txt, tgt in batch:
        src_imgs.append(src)
        src_pats.append(patch)
        txt_feats.append(txt)
        tgt_imgs.append(tgt)

    src_imgs_batched = torch.stack(src_imgs)   # [B, 768]
    src_patches      = torch.stack(src_pats)   # [B, 50, 768]
    tgt_imgs_batched = torch.stack(tgt_imgs)   # [B, 768]

    txt_feats_padded = pad_sequence(txt_feats, batch_first=True, padding_value=0.0)  # [B, max_L, 512]
    text_lengths     = torch.tensor([t.size(0) for t in txt_feats], dtype=torch.long)
    max_len          = txt_feats_padded.size(1)
    batch_size       = len(batch)
    txt_mask         = torch.arange(max_len).expand(batch_size, max_len) < text_lengths.unsqueeze(1)

    return src_imgs_batched, src_patches, txt_feats_padded, txt_mask, text_lengths, tgt_imgs_batched
