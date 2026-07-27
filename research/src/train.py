import os
import sys
import io
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

# Ép Windows Terminal dùng UTF-8 để in tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data.dataset import FashionIQDataset, custom_collate_fn
from models.model import BaselineFusion
from models.loss import CIRLoss

def eval_accuracy(model, dataloader, device):
    """
    Hàm đánh giá Accuracy@1 trên Dev-Subset.
    Mô hình phải được chuyển sang model.eval() trước khi gọi.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src_imgs, txt_feats, txt_mask, txt_lengths, tgt_imgs = batch
            
            src_imgs_cls = src_imgs.to(device)
            txt_feats = txt_feats.to(device)
            txt_lengths = txt_lengths.to(device)
            tgt_imgs_cls = tgt_imgs.to(device)
            
            # Forward
            # Extract txt_eos
            B = src_imgs_cls.size(0)
            eos_indices = txt_lengths - 1
            txt_eos = txt_feats[torch.arange(B, device=device), eos_indices]
            
            combined_query = model(src_imgs_cls, txt_eos)
            q_norm = F.normalize(combined_query, p=2, dim=-1)
            
            target_cls = F.normalize(tgt_imgs_cls, p=2, dim=-1)
            
            # Tính độ tương đồng
            sim_matrix = q_norm @ target_cls.T
            
            # Lấy top-1
            pred = sim_matrix.argmax(dim=1)
            
            # Đếm số lượng đoán đúng (đường chéo)
            correct += (pred == torch.arange(len(pred), device=device)).sum().item()
            total += len(pred)
            
    return (correct / total) * 100.0 if total > 0 else 0.0


def main():
    print("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (DAY 3) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Cấu hình (Config)
    config = {
        "batch_size": 256,
        "epochs": 50,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "temperature": 0.1,
        "dev_size": 300,
        "seed": 42
    }
    
    # Tạo thư mục checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Cố định Seed để chia Data ổn định
    torch.manual_seed(config["seed"])
    
    # 2. Chuẩn bị Dữ liệu
    print("\n[1/4] Đang nạp toàn bộ Dataset (11GB) vào RAM...")
    full_dataset = FashionIQDataset(data_dir="data", category="all")
    
    train_size = len(full_dataset) - config["dev_size"]
    train_subset, dev_subset = random_split(full_dataset, [train_size, config["dev_size"]])
    
    print(f"Tổng mẫu: {len(full_dataset)} | Train: {len(train_subset)} | Dev: {len(dev_subset)}")
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        drop_last=True, 
        collate_fn=custom_collate_fn
    )
    
    dev_loader = DataLoader(
        dev_subset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        drop_last=False, 
        collate_fn=custom_collate_fn
    )
    
    # 3. Khởi tạo Mô hình & Tối ưu hóa
    print("\n2. Khởi tạo Mô hình BaselineFusion...")
    model = BaselineFusion(
        img_dim=768, 
        txt_dim=512, 
        hidden_dim=512, 
        out_dim=768
    ).to(device)
    criterion = CIRLoss(temperature=config["temperature"]).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    # 4. Training Loop
    print(f"\n[3/4] BẮT ĐẦU HUẤN LUYỆN {config['epochs']} EPOCHS...")
    best_dev_acc = 0.0
    
    for epoch in range(1, config["epochs"] + 1):
        # --- TRAIN PHASE ---
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            src_imgs, txt_feats, txt_mask, txt_lengths, tgt_imgs = batch
            
            src_imgs_cls = src_imgs.to(device)
            txt_feats = txt_feats.to(device)
            txt_lengths = txt_lengths.to(device)
            tgt_imgs_cls = tgt_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Extract txt_eos
            B = src_imgs_cls.size(0)
            eos_indices = txt_lengths - 1
            txt_eos = txt_feats[torch.arange(B, device=device), eos_indices]
            
            combined_query = model(src_imgs_cls, txt_eos)
            loss = criterion(combined_query, tgt_imgs_cls)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Step LR Scheduler
        scheduler.step()
        
        # --- EVAL PHASE (Dev Subset) ---
        dev_acc = eval_accuracy(model, dev_loader, device)
        
        print(f"Epoch [{epoch:02d}/{config['epochs']}] | Train Loss: {avg_train_loss:.4f} | Dev Acc@1: {dev_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # --- CHECKPOINTING ---
        # 1. Luôn lưu last
        torch.save(model.state_dict(), "checkpoints/attention_all_last.pth")
        
        # 2. Lưu best nếu Dev Acc tăng
        if dev_acc > best_dev_acc:
            print(f"   >>> Dev Acc tăng từ {best_dev_acc:.2f}% lên {dev_acc:.2f}%. Lưu model tốt nhất tại checkpoints/attention_all_best.pth (Accuracy@1: {dev_acc:.2f}%)")
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), "checkpoints/attention_all_best.pth")

    print(f"\n[4/4] HUẤN LUYỆN HOÀN TẤT! Best Dev Accuracy: {best_dev_acc:.2f}%")
    print("Checkpoints đã được lưu tại thư mục 'research/checkpoints/'")

if __name__ == "__main__":
    main()
