# Fashion-IQ Composed Image Retrieval

Dự án mã nguồn thực nghiệm cho bài toán **Composed Image Retrieval (Truy xuất Ảnh Kết hợp)** trên bộ dữ liệu Fashion-IQ.
Mục tiêu: kết hợp *ảnh gốc* và *câu lệnh ngôn ngữ tự nhiên* để tìm ra bức ảnh đích mong muốn từ kho ảnh 74,381 sản phẩm thời trang.

- **GitHub**: [qwerty12321434/Fashion-Retrieval-System](https://github.com/qwerty12321434/Fashion-Retrieval-System.git)
- **Google Drive (Features)**: [Link Drive](https://drive.google.com/drive/u/0/folders/1p_zyidgXWEOp0k1IaOv1ba8Z9CWVTULi)

---

## 📊 Kết Quả Thực Nghiệm (FashionIQ Validation — 6.016 queries)

Kết quả được tách thành hai protocol, không so sánh trực tiếp điểm giữa hai bảng:

- **FashionIQ Standard:** đánh giá riêng từng category trên đúng validation split; số cuối là macro-average của dress, shirt và toptee.
- **Global-74K:** tất cả query tìm trên toàn bộ 74.381 ảnh local; số cuối là pooled recall trên 6.016 query.

### FashionIQ Standard — FashionCLIP

| Mô hình | Dress R@10/R@50 | Shirt R@10/R@50 | Toptee R@10/R@50 | Macro Avg R@10/R@50 |
|---------|------------------|------------------|-------------------|----------------------|
| Zero-shot Vector Addition | 20.48 / 40.11 | 21.30 / 37.00 | 25.40 / 44.26 | 22.39 / 40.46 |
| BaselineFusion + legacy NT-Xent | 24.64 / 46.06 | 15.90 / 33.42 | 27.08 / 45.95 | 22.54 / 41.81 |
| BaselineFusion + Triplet | 30.04 / 52.70 | 21.69 / 40.87 | 32.23 / 53.03 | 27.99 / 48.87 |
| **BaselineFusion + Batch Classification** | **33.56 / 58.45** | **26.55 / 50.00** | 36.72 / **64.05** | **32.28 / 57.50** |
| AACLFusion + legacy NT-Xent | 32.57 / 54.44 | 22.42 / 41.90 | 33.30 / 55.38 | 29.43 / 50.57 |
| AACLFusion + Triplet | 31.09 / 54.09 | 20.31 / 40.63 | 31.87 / 53.80 | 27.76 / 49.51 |
| AACLFusion + Batch Classification | 33.42 / 56.97 | 25.32 / 49.75 | **37.68** / 62.72 | 32.14 / 56.48 |

Gallery chuẩn lấy trực tiếp từ metadata chính thức: dress 3.817 ảnh, shirt 6.346 ảnh và toptee 5.373 ảnh. File toptee chính thức hiện có 5.373 ASIN, lệch một ảnh so với con số 5.374 trong supplementary.

### Global-74K — FashionCLIP

| Mô hình | Pooled R@10 | Pooled R@50 |
|---------|-------------|-------------|
| Zero-shot Vector Addition | 10.44% | 22.44% |
| BaselineFusion + legacy NT-Xent | 9.41% | 21.23% |
| BaselineFusion + Triplet | 13.08% | 26.65% |
| BaselineFusion + Batch Classification | 12.07% | 27.16% |
| AACLFusion + Triplet | 12.67% | 25.75% |
| AACLFusion + Batch Classification | 12.77% | 26.93% |
| **AACLFusion + legacy NT-Xent** | **13.93%** | **27.88%** |

> **Key Insight:** Kết luận phụ thuộc protocol. Batch Classification tốt nhất trên FashionIQ Standard, trong đó BaselineFusion đạt macro R@10/R@50 = **32.28/57.50**. Ngược lại, AACLFusion + legacy NT-Xent tốt nhất trên Global-74K với pooled R@10/R@50 = **13.93/27.88**. Vì vậy, Global-74K được xem là thí nghiệm catalog bổ sung và không dùng để so trực tiếp với kết quả paper.

> **Rank policy:** Rank một-based được tính bằng `1 + count(similarity > target_similarity)`, tương đương chính sách strict-distance của evaluator tham chiếu. Exact-score ties đã được audit trên toàn bộ checkpoint trước khi đóng băng bảng.

### CLIP-base — Global-74K (thử nghiệm backbone ban đầu)

| Mô hình | Recall@10 | Recall@50 |
|---------|-----------|-----------|
| Zero-shot Vector Addition | 5.82% | 13.61% |
| BaselineFusion + InfoNCE | 6.78% | 17.07% |
| BaselineFusion + Triplet | 6.52% | 16.37% |

> **Giới hạn thống kê:** Mỗi cấu hình trong bảng hiện mới được chạy với một seed (`42`). Các chênh lệch hiện tại là kết quả thực nghiệm ban đầu, chưa phải kết luận về độ ổn định qua nhiều seed.

---

## 📂 Cấu Trúc Project

```
FashionSystem/
└── research/
    ├── src/
    │   ├── train.py                        # Script huấn luyện
    │   ├── evaluate.py                     # Script đánh giá Recall@10/50
    │   ├── demo.py                         # Script demo trực quan
    │   ├── data/
    │   │   └── dataset.py                  # FashionIQDataset, CategoryBatchSampler
    │   └── models/
    │       ├── model.py                    # BaselineFusion, AdditiveAttention
    │       └── loss.py                     # InfoNCE, Triplet, Batch Classification
    ├── scripts/
    │   ├── extract_features.py             # Trích xuất CLIP-base/FashionCLIP
    │   └── prep_candidate_patches.py       # Lọc candidate token cho AACL
    ├── data/
    │   ├── json/                           # FashionIQ annotation files
    │   ├── features/                       # Features từ CLIP-base
    │   │   ├── gallery_cls_768.pt          # [74381, 768] CLS image features
    │   │   ├── gallery_embeds_512.pt       # [74381, 512] Projected embeddings
    │   │   ├── gallery_asins.json
    │   │   ├── {dress|shirt|toptee}_text_tokens.pt
    │   │   ├── {cat}_val_text_hidden.pt
    │   │   └── {cat}_val_text_embeds.pt
    │   └── features_fashionclip/           # Features từ FashionCLIP (cùng cấu trúc)
    ├── checkpoints/                        # Checkpoint .pth + config .json
    └── requirements.txt
```

---

## 🧠 Kiến Trúc Mô Hình

### Backbone
| Tên | Model ID | Ghi chú |
|-----|----------|---------|
| CLIP-base | `openai/clip-vit-base-patch32` | Tổng quát, dùng làm baseline |
| **FashionCLIP** | `patrickjohncyh/fashion-clip` | Fine-tuned thời trang, **khuyên dùng** |

Cả hai đều dùng kiến trúc **ViT-B/32**: image token `[50, 768]`, text hidden `[seq_len, 512]`, projected embed `[512]`.

### Pipeline
```
Candidate Image ──[Backbone Encoder]──> CLS [768]  ─┐
                                                      ├─> BaselineFusion (MLP) ──> Query [768]
Text Modifier   ──[Backbone Encoder]──> EOS [512]  ─┘

Query [768]  ──[Cosine Similarity]──> Gallery [74381, 768] ──> Top-K Results
```

### Loss Functions
| Loss | Class | Mô tả |
|------|-------|-------|
| InfoNCE | `CIRLoss` | Phạt tất cả negatives trong batch — hiệu quả khi không gian embedding hỗn loạn |
| Triplet | `CIRTripletLoss` | Chỉ phạt negative quá gần positive — tốt hơn khi không gian đã có cấu trúc (FashionCLIP) |
| Batch Classification | `CIRBatchClassificationLoss` | Loss một chiều query→target theo AACL paper; softmax trên toàn bộ target trong batch |

InfoNCE và Triplet hiện ghép query với target vào cùng metric-learning objective, trong khi Batch Classification paper-faithful chỉ tối ưu chiều query→target. Vì vậy đây là ba chiến lược tối ưu khác nhau trên cùng protocol CIR, không phải ba objective hoàn toàn tương đương.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 0. Yêu Cầu Hệ Thống
- GPU NVIDIA với CUDA 12.1+
- RAM ≥ 16GB (`candidate_patch_tokens.pt` hiện khoảng 2,41 GiB trên disk; khi load/train còn có tensor và DataLoader overhead)

### 1. Cài Đặt Môi Trường

```bash
git clone https://github.com/qwerty12321434/Fashion-Retrieval-System.git
cd FashionSystem

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers pytorch-metric-learning pillow tqdm matplotlib
```

### 2. Chuẩn Bị Dữ Liệu

**Tải features từ Google Drive** (nếu không muốn extract lại) → đặt vào `research/data/features/`.

**Hoặc tự extract** (yêu cầu thư mục ảnh FashionIQ):

```bash
cd research

# FashionCLIP features (lần đầu download ~600MB, khuyên dùng)
python scripts/extract_features.py --backbone fashionclip
# Output: data/features_fashionclip/

# CLIP-base features (pipeline cũ, để so sánh)
python scripts/extract_features.py --backbone clip-base
# Output: data/features/

# (Chỉ cần 1 lần) Lọc visual tokens cho AACL — bắt buộc trước khi train --arch aacl
# Input: all_image_tokens.pt (~11GB) | Output đã lọc: candidate_patch_tokens.pt (~2.41GiB)
# Mỗi ảnh có 50 visual tokens: 1 CLS + 49 spatial patches.
python scripts/prep_candidate_patches.py
```

### 3. Huấn Luyện

```bash
cd research

# === CLIP-base ===
python src/train.py --run_name baseline_infonce_1024 --loss infonce --epochs 50
python src/train.py --run_name baseline_triplet_1024 --loss triplet --epochs 50

# === FashionCLIP (khuyên dùng) ===
python src/train.py --run_name fashionclip_infonce_1024 --loss infonce --features_dir data/features_fashionclip
python src/train.py --run_name fashionclip_triplet_1024 --loss triplet  --features_dir data/features_fashionclip
python src/train.py --run_name fashionclip_batchcls_1024 --loss batch_cls --features_dir data/features_fashionclip

# === AACL ===
python src/train.py --arch aacl --run_name aacl_fashionclip_infonce --loss infonce --features_dir data/features_fashionclip --epochs 50
python src/train.py --arch aacl --run_name aacl_fashionclip_triplet --loss triplet --features_dir data/features_fashionclip --epochs 50
python src/train.py --arch aacl --run_name aacl_fashionclip_batchcls --loss batch_cls --features_dir data/features_fashionclip --epochs 50

# Seed của model/batch shuffle tách khỏi seed chia dev; dev mặc định cân bằng 100 mẫu/category.
python src/train.py --arch aacl --loss infonce --seed 7 --split_seed 42 --dev_per_category 100
```

Checkpoint tốt nhất tự động lưu tại `checkpoints/{run_name}_best.pth`.

### 4. Đánh Giá

`evaluate.py` mặc định dùng protocol `fashioniq`. Dùng `--protocol global` khi cần tái hiện bảng Global-74K. Có thể thêm `--output_json` để lưu metric có cấu trúc trong `reports/`.

```bash
cd research

# === FashionIQ Standard: BaselineFusion (MLP: CLS + EOS) ===
python src/evaluate.py \
  --protocol fashioniq \
  --ckpt fashionclip_infonce_1024_best.pth fashionclip_triplet_1024_best.pth fashionclip_batchcls_1024_best.pth \
  --features_dir data/features_fashionclip \
  --output_json reports/eval_baseline_fashioniq.json

# === FashionIQ Standard: AACLFusion ===
python src/evaluate.py --arch aacl \
  --protocol fashioniq \
  --ckpt aacl_fashionclip_infonce_best.pth aacl_fashionclip_triplet_best.pth aacl_fashionclip_batchcls_best.pth \
  --features_dir data/features_fashionclip \
  --output_json reports/eval_aacl_fashioniq.json

# === Global-74K: thêm --protocol global ===
python src/evaluate.py --arch aacl \
  --protocol global \
  --ckpt aacl_fashionclip_infonce_best.pth \
  --features_dir data/features_fashionclip \
  --output_json reports/eval_aacl_global.json
```

### 5. Demo Trực Quan

`demo.py` hỗ trợ hai chế độ query và hai phạm vi gallery:

- **Validation offline:** tự lấy candidate, modifier và target thật từ FashionIQ; không cần tải backbone.
- **Free-form:** nhập ASIN và modifier tùy ý; cần backbone Hugging Face đã cache hoặc có kết nối mạng.
- **`--gallery_scope global` (mặc định):** tìm trên toàn bộ 74.381 ảnh.
- **`--gallery_scope category`:** tìm trên đúng validation split của category; chỉ dùng cùng `--val_index`.

```bash
cd research

# === Validation offline: FashionIQ category gallery ===
python src/demo.py \
  --category dress --gallery_scope category --val_index 1 \
  --ckpt fashionclip_batchcls_1024_best.pth \
  --arch baseline --top_k 5 \
  --output demo_baseline_batchcls.png

# === Validation offline: Global-74K gallery ===
python src/demo.py \
  --category dress --gallery_scope global --val_index 1 \
  --ckpt aacl_fashionclip_batchcls_best.pth \
  --arch aacl --top_k 5 \
  --output demo_aacl_batchcls.png

# === Free-form: BaselineFusion + Triplet ===
python src/demo.py \
  --candidate "B00FHFMMMW" \
  --text "is softly colored, has no shoulder straps and looser skirt" \
  --ckpt fashionclip_triplet_1024_best.pth \
  --output demo_baseline_triplet.png

# === Free-form: AACL-inspired + InfoNCE ===
python src/demo.py --arch aacl \
  --candidate "B00FHFMMMW" \
  --text "is softly colored, has no shoulder straps and looser skirt" \
  --ckpt aacl_fashionclip_infonce_best.pth \
  --output demo_aacl_infonce.png

# Nếu biết target thật trong free-form mode:
# thêm --target TARGET_ASIN để tính rank và highlight.
```

---

## ⚠️ Lưu Ý Quan Trọng

| Vấn đề | Giải thích |
|--------|-----------|
| **Đồng bộ backbone & features_dir** | Checkpoint được train với backbone nào thì phải dùng `--features_dir` tương ứng. Sai sẽ cho kết quả vô nghĩa nhưng không crash. |
| **Chạy từ `research/`** | Tất cả lệnh chạy từ `FashionSystem/research/` để đường dẫn tương đối (`data/`, `checkpoints/`, `src/`) hoạt động đúng. |
| **Windows + DataLoader** | Script có `if __name__ == "__main__":` guard để tránh deadlock với `num_workers > 0` trên Windows. |

---

## 📈 Lộ Trình Phát Triển

| Mức | Tính năng | Trạng thái |
|-----|-----------|-----------|
| 0 | Fix tên checkpoint động (`--run_name`) | ✅ Hoàn thành |
| 1 | Triplet Loss + arg `--loss` | ✅ Hoàn thành |
| 2 | AttentionFusion (50 visual tokens: 1 CLS + 49 patches) | 🔄 Đang lên kế hoạch |
| 3 | Attribute Classifier + Rerank | ⬜ Chưa bắt đầu |
| 4 | FashionCLIP backbone (`--backbone`, `--features_dir`) | ✅ Hoàn thành |
