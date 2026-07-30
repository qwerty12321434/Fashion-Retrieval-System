# Fashion-IQ Composed Image Retrieval

Dự án mã nguồn thực nghiệm cho bài toán **Composed Image Retrieval (Truy xuất Ảnh Kết hợp)** trên bộ dữ liệu Fashion-IQ.
Mục tiêu: kết hợp *ảnh gốc* và *câu lệnh ngôn ngữ tự nhiên* để tìm ra bức ảnh đích mong muốn từ kho ảnh 74,381 sản phẩm thời trang.

- **GitHub**: [qwerty12321434/Fashion-Retrieval-System](https://github.com/qwerty12321434/Fashion-Retrieval-System.git)
- **Google Drive (Features)**: [Link Drive](https://drive.google.com/drive/u/0/folders/1p_zyidgXWEOp0k1IaOv1ba8Z9CWVTULi)

---

## 📊 Kết Quả Thực Nghiệm (FashionIQ Validation — 6016 queries)

### Backbone: CLIP-base (`openai/clip-vit-base-patch32`)
| Mô hình | Recall@10 | Recall@50 | Checkpoint |
|---------|-----------|-----------|-----------|
| Zero-shot Vector Addition | 5.82% | 13.61% | — |
| BaselineFusion + InfoNCE | 6.78% | 17.07% | `baseline_infonce_1024_best.pth` |
| BaselineFusion + Triplet | 6.52% | 16.37% | `baseline_triplet_1024_best.pth` |

### Backbone: FashionCLIP (`patrickjohncyh/fashion-clip`) ⭐ Best
| Mô hình | Kiến trúc | R@10 | R@50 | Checkpoint |
|---------|-----------|------|------|------------|
| Zero-shot Vector Addition | — | 10.32% | 22.39% | — |
| BaselineFusion + InfoNCE | MLP (CLS+EOS) | 9.18% | 21.14% | `fashionclip_infonce_1024_best.pth` |
| BaselineFusion + Triplet | MLP (CLS+EOS) | 12.88% | 26.56% | `fashionclip_triplet_1024_best.pth` |
| BaselineFusion + Batch Classification | MLP (CLS+EOS) | 11.88% | 27.11% | `fashionclip_batchcls_1024_best.pth` |
| AACLFusion + Triplet | Additive Attention (50 patches + full text) | 12.45% | 25.66% | `aacl_fashionclip_triplet_best.pth` |
| AACLFusion + Batch Classification | Additive Attention (50 tokens + full text) | 12.53% | 26.88% | `aacl_fashionclip_batchcls_best.pth` |
| **AACLFusion + InfoNCE** | Additive Attention (50 patches + full text) | **13.76%** | **27.78%** | `aacl_fashionclip_infonce_best.pth` |

> **Key Insight (Phase B):** Biến thể AACL-inspired single-head với InfoNCE vẫn cho kết quả cao nhất (**R@10: 13.76%, R@50: 27.78%**). Batch Classification một chiều theo công thức AACL paper cải thiện BaselineFusion so với InfoNCE và đạt R@50 cao nhất trong ba loss của Baseline, nhưng chưa vượt Triplet ở R@10 và chưa vượt InfoNCE trên AACLFusion. Điều này cho thấy loss gốc của paper không tự động chuyển thành loss tốt nhất khi dùng frozen FashionCLIP và kiến trúc AACL rút gọn.

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

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 0. Yêu Cầu Hệ Thống
- GPU NVIDIA với CUDA 12.1+
- RAM ≥ 16GB (gallery features chiếm ~1.1GB VRAM + ~11GB RAM khi train patch tokens)

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

# (Chỉ cần 1 lần) Lọc patch tokens cho AACL — bắt buộc trước khi train --arch aacl
# Input: all_image_tokens.pt (~11GB) | Output dự kiến sau clone: candidate_patch_tokens.pt (~2.5GB)
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
```

Checkpoint tốt nhất tự động lưu tại `checkpoints/{run_name}_best.pth`.

### 4. Đánh Giá

```bash
cd research

# === BaselineFusion (MLP: CLS + EOS) ===
python src/evaluate.py \
  --ckpt fashionclip_infonce_1024_best.pth fashionclip_triplet_1024_best.pth fashionclip_batchcls_1024_best.pth \
  --features_dir data/features_fashionclip

# === AACLFusion (Additive Attention: 50 patches + full text) ===
python src/evaluate.py --arch aacl \
  --ckpt aacl_fashionclip_infonce_best.pth aacl_fashionclip_triplet_best.pth aacl_fashionclip_batchcls_best.pth \
  --features_dir data/features_fashionclip
```

### 5. Demo Trực Quan

`demo.py` hỗ trợ hai chế độ:

- **Validation offline:** tự lấy candidate, modifier và target thật từ FashionIQ; không cần tải backbone. Ảnh output hiển thị full-gallery rank và viền xanh target.
- **Free-form:** nhập ASIN và modifier tùy ý; cần backbone Hugging Face đã cache hoặc có kết nối mạng.

```bash
cd research

# === Validation offline: Baseline + Batch Classification ===
python src/demo.py \
  --category dress --val_index 1 \
  --ckpt fashionclip_batchcls_1024_best.pth \
  --arch baseline --top_k 5 \
  --output demo_baseline_batchcls.png

# === Validation offline: AACL-inspired + Batch Classification ===
python src/demo.py \
  --category dress --val_index 1 \
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
| 2 | AttentionFusion (50 patch tokens) | 🔄 Đang lên kế hoạch |
| 3 | Attribute Classifier + Rerank | ⬜ Chưa bắt đầu |
| 4 | FashionCLIP backbone (`--backbone`, `--features_dir`) | ✅ Hoàn thành |
