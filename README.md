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
| Mô hình | Recall@10 | Recall@50 | Checkpoint |
|---------|-----------|-----------|-----------|
| Zero-shot Vector Addition | 10.32% | 22.39% | — |
| BaselineFusion + InfoNCE | 9.18% | 21.13% | `fashionclip_infonce_1024_best.pth` |
| **BaselineFusion + Triplet** | **12.88%** | **26.56%** | `fashionclip_triplet_1024_best.pth` |

> **Key Insight:** FashionCLIP (fine-tuned trên ~70K sản phẩm thời trang) kết hợp Triplet Loss đạt hiệu năng tốt nhất. FashionCLIP tạo ra không gian embedding đã có cấu trúc tốt cho domain thời trang, nên Triplet Loss (với margin) bảo toàn cấu trúc đó tốt hơn InfoNCE vốn đẩy mạnh toàn bộ negatives ra xa.

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
    │       └── loss.py                     # CIRLoss (InfoNCE), CIRTripletLoss
    ├── scripts/
    │   ├── extract_features.py             # Trích xuất feature bằng CLIP-base
    │   └── extract_features_fashionclip.py # Trích xuất feature bằng FashionCLIP ⭐
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

# CLIP-base features (pipeline cũ)
python scripts/extract_features.py

# FashionCLIP features (lần đầu download ~600MB, khuyên dùng)
python scripts/extract_features_fashionclip.py
# Output: data/features_fashionclip/
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
```

Checkpoint tốt nhất tự động lưu tại `checkpoints/{run_name}_best.pth`.

### 4. Đánh Giá

```bash
cd research

# So sánh nhiều checkpoint cùng lúc (cùng features_dir)
python src/evaluate.py \
  --ckpt fashionclip_infonce_1024_best.pth fashionclip_triplet_1024_best.pth \
  --features_dir data/features_fashionclip
```

### 5. Demo Trực Quan

```bash
cd research

# Demo với CLIP-base
python src/demo.py \
  --candidate "B00FHFMMMW" \
  --text "is softly colored, has no shoulder straps and looser skirt" \
  --ckpt baseline_infonce_1024_best.pth \
  --output demo_result.png

# Demo với FashionCLIP (backbone và features_dir phải đồng bộ với checkpoint!)
python src/demo.py \
  --candidate "B00FHFMMMW" \
  --text "is softly colored, has no shoulder straps and looser skirt" \
  --ckpt fashionclip_triplet_1024_best.pth \
  --backbone patrickjohncyh/fashion-clip \
  --features_dir data/features_fashionclip \
  --output demo_fashionclip.png
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