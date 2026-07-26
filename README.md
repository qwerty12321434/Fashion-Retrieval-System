# Fashion-IQ Composed Image Retrieval

Dự án này là mã nguồn thực nghiệm cho bài toán **Composed Image Retrieval (Truy xuất ảnh kết hợp)** trên bộ dữ liệu Fashion-IQ. Mục tiêu là kết hợp ảnh gốc và câu lệnh ngôn ngữ tự nhiên để tìm ra bức ảnh đích.

## 📌 Yêu cầu hệ thống (Quan trọng)
- Hệ điều hành: Windows hoặc Linux (Khuyến nghị có GPU NVIDIA).
- **Python: Chỉ sử dụng Python 3.11 hoặc 3.12** (Tuyệt đối KHÔNG dùng Python 3.13 vì hiện tại PyTorch chưa có bản cài đặt CUDA hỗ trợ GPU cho phiên bản này, mô hình sẽ bị ép chạy bằng CPU rất chậm).

---

## 🚀 Hướng dẫn Cài đặt & Chạy Dự án

### Bước 1: Clone mã nguồn
Tải mã nguồn dự án về máy tính của bạn:
```bash
git clone <https://github.com/qwerty12321434/Fashion-Retrieval-System.git>
cd FashionSystem
```

### Bước 2: Thiết lập Môi trường Ảo (Virtual Environment)
Môi trường ảo giúp các thư viện của dự án này không bị xung đột với các dự án khác trên máy bạn.

**Mở Terminal (PowerShell/CMD) và chạy lệnh:**
```bash
# Tạo môi trường ảo có tên là 'venv'
python -m venv venv

# Kích hoạt môi trường ảo (Dành cho Windows PowerShell)
.\venv\Scripts\Activate.ps1

# (Nếu dùng Linux hoặc MacOS, chạy lệnh này thay thế: source venv/bin/activate)
```
*(Nếu Windows báo lỗi không cho chạy script `Activate.ps1`, hãy mở PowerShell dưới quyền Admin và chạy lệnh `Set-ExecutionPolicy Unrestricted -Force`, sau đó kích hoạt lại).*

### Bước 3: Cài đặt Thư viện & PyTorch (Bản GPU)
Chúng ta cần cài đặt PyTorch phiên bản hỗ trợ nhân CUDA (để tận dụng sức mạnh của Card đồ họa NVIDIA).

Đảm bảo bạn vẫn đang ở trong môi trường `(venv)`, chạy lệnh sau:
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow tqdm
```
*(Quá trình này có thể mất 5-10 phút để tải file PyTorch nặng khoảng 2.5GB).*

### Bước 4: Tải Dữ liệu (Features & Data)
Vì các file đặc trưng (Tensors) rất nặng (khoảng 11GB) nên không được đẩy lên GitHub. Bạn cần tải chúng từ Google Drive.

1. Tải file nén dữ liệu từ link Drive: [https://drive.google.com/drive/u/0/folders/1p_zyidgXWEOp0k1IaOv1ba8Z9CWVTULi]
2. Giải nén và đặt các file vào đúng cấu trúc thư mục sau:
   ```text
   FashionSystem/
   ├── research/
   │   ├── data/
   │   │   ├── features/
   │   │   │   ├── all_image_tokens.pt       # (Đặc trưng của 74k ảnh)
   │   │   │   ├── dress_text_tokens.pt      
   │   │   │   ├── shirt_text_tokens.pt
   │   │   │   └── toptee_text_tokens.pt
   │   │   ├── json/
   │   │   │   ├── cap.dress.train.json      # (File caption gốc)
   │   │   │   ├── cap.shirt.train.json
   │   │   │   └── cap.toptee.train.json
   ```

### Bước 5: Chạy Script Trích xuất (Dành cho Test)
Nếu bạn muốn tự mình chạy lại quá trình trích xuất để kiểm tra hệ thống đã nhận GPU hay chưa:

```bash
cd research
python scripts/extract_features.py
```
*(Script sẽ tự động đọc file JSON, trích xuất mã hóa bằng mô hình CLIP và lưu thành các file `.pt` trong thư mục `features`).*

---

## 📝 Nhật ký Phát triển
- 1: Hoàn thiện tiền xử lý dữ liệu, DataLoader và trích xuất thành công đặc trưng CLIP (Vision & Text) ra dạng `.pt` siêu nhẹ để tăng tốc độ huấn luyện.
- 2: (Đang cập nhật) - Xây dựng mô đun Additive Attention Fusion và hàm Loss Hard-Negative.