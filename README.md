# Fashion-IQ Composed Image Retrieval

Dự án này là mã nguồn thực nghiệm cho bài toán **Composed Image Retrieval (Truy xuất ảnh kết hợp)** trên bộ dữ liệu Fashion-IQ. 
Mục tiêu là kết hợp ảnh gốc và câu lệnh ngôn ngữ tự nhiên để tìm ra bức ảnh đích mong muốn.

- **GitHub Repository**: [https://github.com/qwerty12321434/Fashion-Retrieval-System.git](https://github.com/qwerty12321434/Fashion-Retrieval-System.git)
- **Google Drive Dữ liệu (Features)**: [https://drive.google.com/drive/u/0/folders/1p_zyidgXWEOp0k1IaOv1ba8Z9CWVTULi](https://drive.google.com/drive/u/0/folders/1p_zyidgXWEOp0k1IaOv1ba8Z9CWVTULi)

## 📌 Bộ Dữ Liệu Fashion-IQ
Fashion-IQ là một bộ dữ liệu chuẩn mực cho việc tìm kiếm ảnh thời trang dựa trên ngôn ngữ tự nhiên. 
Nó chứa 3 danh mục chính:
- **Dress (Váy)**
- **Shirt (Áo sơ mi)**
- **Top & Tee (Áo thun)**

Mỗi truy vấn (query) trong bộ dữ liệu bao gồm: một **ảnh gốc (candidate image)** và hai **câu lệnh chỉnh sửa (modifiers)** mô tả sự khác biệt về mặt thiết kế/màu sắc giữa ảnh gốc và **ảnh mục tiêu (target image)**. 

---

## 🚀 Hướng dẫn Cài đặt Môi trường

Hệ thống được thiết kế chạy tối ưu nhất với GPU NVIDIA (CUDA).

### Bước 1: Clone mã nguồn
```bash
git clone https://github.com/qwerty12321434/Fashion-Retrieval-System.git
cd FashionSystem
```

### Bước 2: Thiết lập Môi trường Ảo
```bash
python -m venv venv
# Kích hoạt trên Windows:
.\venv\Scripts\Activate.ps1
# Kích hoạt trên Linux/MacOS:
source venv/bin/activate
```

### Bước 3: Cài đặt Thư viện
Dự án sử dụng các thư viện cốt lõi sau:
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `transformers==5.14.1`
- `pytorch-metric-learning`
- `tqdm`, `pillow`, `matplotlib`

Chạy lệnh sau để cài đặt toàn bộ:
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers pytorch-metric-learning pillow tqdm matplotlib
```

### Bước 4: Chuẩn bị Dữ liệu
1. Tải toàn bộ thư mục `features/` và `json/` từ Google Drive ở trên.
2. Đặt vào đúng cấu trúc sau:
   ```text
   FashionSystem/
   ├── research/
   │   ├── data/
   │   │   ├── features/
   │   │   │   ├── gallery_cls_768.pt      # Đặc trưng của 74k ảnh (1.1GB)
   │   │   │   ├── gallery_embeds_512.pt   # (Cho Baseline CLIP Zero-shot)
   │   │   │   ├── dress_text_tokens.pt    # Đặc trưng văn bản
   │   │   │   ├── shirt_text_tokens.pt
   │   │   │   └── toptee_text_tokens.pt
   │   │   ├── json/
   │   │   │   ├── cap.dress.train.json
   │   │   │   ├── cap.shirt.train.json
   │   │   │   └── cap.toptee.train.json
   ```

---

## 🛠️ Quá trình Thực hiện & Cách chạy Script

Dự án được phân chia thành các thư mục rõ ràng:
- `data/`: Chứa Dataset Loader và các script tiền xử lý/trích xuất đặc trưng. Sử dụng kỹ thuật `CategoryBatchSampler` (Hard Negative) để ép mô hình học đặc trưng chi tiết.
- `models/`: Chứa định nghĩa kiến trúc mạng AI (`model.py`) và các hàm mất mát (`loss.py` gồm `InfoNCE` và `TripletLoss`).
- `checkpoints/`: Nơi lưu trữ tự động các trọng số (`.pth`) và file cấu hình (`_config.json`) của từng thực nghiệm.

Tất cả các script thực thi chính đều nằm trong `research/src/`. Dưới đây là luồng chạy (pipeline) tiêu chuẩn:

### Giai đoạn 1: Trích xuất Đặc trưng (Feature Extraction)
Sử dụng mô hình pre-trained `CLIP (openai/clip-vit-base-patch32)` đã đóng băng (freeze encoder) để chuyển đổi hình ảnh và văn bản thành các vector (`.pt`). Quá trình này giúp tối ưu hóa RAM tuyệt đối và đẩy nhanh tốc độ huấn luyện.

- **Cách chạy:**
  ```bash
  cd research
  python src/data/extract_val.py
  python src/data/prep_gallery.py
  ```

### Giai đoạn 2: Huấn luyện (Training)
Chúng tôi sử dụng kiến trúc **BaselineFusion**: 
Nhận `CLS Token` của ảnh gốc (vector 768 chiều) và `EOS Token` của văn bản (512 chiều) -> Nối lại (Concat) -> Cho qua mạng MLP 2 lớp (với `hidden_dim=1024`) -> So sánh với `CLS Token` của ảnh đích.

Đặc biệt, DataLoader sử dụng **Hard Negative Sampling** (đảm bảo 1 batch chứa toàn các mẫu cùng loại) để tăng độ khó, kết hợp với hàm **InfoNCE Loss** để tối ưu.

- **Cách chạy:**
  Script hỗ trợ các tham số dòng lệnh (`argparse`) để dễ dàng quản lý thực nghiệm và chọn hàm loss (`infonce` hoặc `triplet`):
  ```bash
  cd research
  python src/train.py --run_name baseline_infonce_1024 --loss infonce --epochs 50
  ```
- **Kết quả:** Hệ thống tự động tạo file `checkpoints/baseline_infonce_1024_config.json` và lưu trọng số tốt nhất tại `checkpoints/baseline_infonce_1024_best.pth`.

### Giai đoạn 3: Đánh giá (Evaluation)
Đo lường độ chính xác trên tập Validation của toàn bộ 3 danh mục bằng chỉ số Recall@10 và Recall@50. Script hỗ trợ **đánh giá nhiều mô hình cùng lúc** (`nargs='+'`).

- **Cách chạy:**
  ```bash
  cd research
  python src/evaluate.py --ckpt baseline_infonce_1024_best.pth baseline_triplet_1024_best.pth
  ```

- **Kết quả ghi nhận:**
  ```text
  === KẾT QUẢ TỔNG HỢP ===
  Tổng số truy vấn hợp lệ: 6016/6016

  1. CLIP ZERO-SHOT (Vector Addition)
   - Recall@10: 5.82%
   - Recall@50: 13.61%

  2. BASELINE FUSION (baseline_infonce_1024)
   - Recall@10: 6.78%
   - Recall@50: 17.07%

  3. BASELINE FUSION (baseline_triplet_1024)
   - Recall@10: 6.52%
   - Recall@50: 16.37%
  ```

### Giai đoạn 4: Trực quan hóa (Demo)
Script cho phép nhập 1 ASIN ảnh gốc (Candidate) và 1 câu lệnh modifier để tìm ra top 5 kết quả tốt nhất. Nó sẽ vẽ biểu đồ so sánh trực quan giữa Zero-Shot và mô hình đã train (tự động phát hiện kích thước kiến trúc từ file weights).

- **Cách chạy:**
  ```bash
  cd research
  python src/demo.py --candidate "B00FHFMMMW" --text "is softly colored,has no shoulder straps and looser skirt" --ckpt baseline_infonce_1024_best.pth --output "demo_result.png"
  ```
- **Kết quả:**
  ![Demo Result](demo_result.png)

---

## 🎯 Kết luận và Hướng Phát Triển
Dự án đã trải qua nhiều pha tối ưu hóa:
1. Nâng cấp bộ nhớ mạng (MLP `hidden_dim` từ 512 lên 1024).
2. Chuyển đổi chiến lược học từ ngẫu nhiên (Random Sampling) sang thử thách (Hard Negative Sampling).
3. Chứng minh hàm đối chiếu InfoNCE hiệu quả hơn Triplet Loss với batch_size lớn.

**Tuy nhiên, các kết quả hiện tại chỉ ra một Nút Thắt Cổ Chai (Bottleneck) về kiến trúc:** 
Mạng `BaselineFusion` hiện tại chỉ nối trực tiếp vector tổng quát (Global CLS Token) của ảnh và chữ. Cách nén thông tin thô bạo này làm mất đi toàn bộ chi tiết không gian (spatial details) của bức ảnh (như vị trí cổ áo, họa tiết logo). Việc tăng tham số mạng hay làm khó bộ nạp dữ liệu không thể vượt qua ngưỡng giới hạn này.

**Hướng đi tiếp theo (Giai đoạn 2):** 
Dự án sẽ chuyển sang phát triển cấu trúc **Attention Fusion**. Thay vì dùng CLS Token, chúng tôi sẽ nạp 49 Patch Tokens của bức ảnh, dùng câu chữ (Text EOS) để làm Query chiếu (Attend) vào từng mảnh ghép của ảnh. Từ đó giúp mô hình thực sự "hiểu" được câu chữ đang nhắm tới thay đổi phần nào trên bức ảnh!