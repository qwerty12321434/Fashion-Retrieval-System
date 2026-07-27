import torch

# 1. Chỉ định đường dẫn tới file bạn muốn xem
# Bạn có thể đổi thành "data/features/shirt_text_tokens.pt" để xem chữ
file_path = "data/features/shirt_text_tokens.pt" 

print(f"Đang đọc file: {file_path}...\n")

# 2. Tải dữ liệu từ file .pt
data = torch.load(file_path)



# 3. Khám phá cấu trúc tổng thể
print(f"Kiểu dữ liệu tổng thể: {type(data)}")
print(f"Tổng số phần tử (số lượng ảnh/text): {len(data)}\n")

# Lấy thử 3 mã ID đầu tiên ra xem
keys = list(data.keys())[:3]
print(f"3 mã ID đầu tiên trong file: {keys}\n")

# 4. Trích xuất thử MỘT bức ảnh (hoặc 1 câu text) để xem các con số
first_id = keys[0]
tensor_data = data[first_id]

print(data[first_id].device)

print(f"--- ĐANG XEM DỮ LIỆU CỦA ID: {first_id} ---")
print(f"Kích thước (Shape) của ma trận: {tensor_data.shape}")
print("(Giải thích: Kích thước này thường là [Số lượng tokens/ô vuông, 512 chiều])\n")

print("Dưới đây là hình hài thực tế của 5 con số đầu tiên thuộc token đầu tiên:")
# In ra dòng đầu tiên (token số 0), và lấy 5 giá trị đầu tiên của dòng đó
print(tensor_data[0][:5])