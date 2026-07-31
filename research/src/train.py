import os
import sys
import io
import json
import argparse
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

# Ép Windows Terminal dùng UTF-8 để in tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data.dataset import FashionIQDataset, custom_collate_fn, aacl_collate_fn, CategoryBatchSampler
from models.model import BaselineFusion, AACLFusion
from models.loss import CIRLoss, CIRTripletLoss, CIRBatchClassificationLoss


def stratified_train_dev_split(dataset, dev_per_category, split_seed):
    """Tạo dev split cân bằng category và độc lập với training seed."""
    if dev_per_category <= 0:
        raise ValueError("dev_per_category phải lớn hơn 0")

    category_indices = {}
    for index, item in enumerate(dataset.data):
        category = item.get("category")
        if category is None:
            raise ValueError(f"Mẫu index={index} không có trường category")
        category_indices.setdefault(category, []).append(index)

    rng = random.Random(split_seed)
    train_indices = []
    dev_indices = []
    dev_counts = {}

    for category in sorted(category_indices):
        indices = category_indices[category].copy()
        if len(indices) <= dev_per_category:
            raise ValueError(
                f"Category {category} chỉ có {len(indices)} mẫu, không đủ "
                f"để lấy {dev_per_category} dev samples"
            )
        rng.shuffle(indices)
        dev_indices.extend(indices[:dev_per_category])
        train_indices.extend(indices[dev_per_category:])
        dev_counts[category] = dev_per_category

    # Tránh để dev loader nhận một dải category cố định theo thứ tự.
    rng.shuffle(train_indices)
    rng.shuffle(dev_indices)
    return (
        Subset(dataset, train_indices),
        Subset(dataset, dev_indices),
        dev_counts,
    )


def eval_accuracy(model, dataloader, device, arch="baseline"):
    """
    Hàm đánh giá Accuracy@1 trên Dev-Subset.
    Mô hình phải được chuyển sang model.eval() trước khi gọi.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if arch == "aacl":
                src_imgs, src_patches, txt_feats, txt_mask, txt_lengths, tgt_imgs = batch
                src_imgs    = src_imgs.to(device)
                src_patches = src_patches.to(device)
                txt_feats   = txt_feats.to(device)
                txt_mask    = txt_mask.to(device)
                tgt_imgs    = tgt_imgs.to(device)
                combined_query = model(src_patches, txt_feats, txt_mask)
            else:
                src_imgs, txt_feats, txt_mask, txt_lengths, tgt_imgs = batch
                src_imgs_cls = src_imgs.to(device)
                txt_feats    = txt_feats.to(device)
                txt_lengths  = txt_lengths.to(device)
                tgt_imgs     = tgt_imgs.to(device)
                B            = src_imgs_cls.size(0)
                eos_indices  = txt_lengths - 1
                txt_eos      = txt_feats[torch.arange(B, device=device), eos_indices]
                combined_query = model(src_imgs_cls, txt_eos)
            
            q_norm      = F.normalize(combined_query, p=2, dim=-1)
            target_cls  = F.normalize(tgt_imgs, p=2, dim=-1)
            sim_matrix  = q_norm @ target_cls.T
            pred        = sim_matrix.argmax(dim=1)
            correct    += (pred == torch.arange(len(pred), device=device)).sum().item()
            total      += len(pred)
            
    return (correct / total) * 100.0 if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train BaselineFusion / AACLFusion model")
    parser.add_argument("--run_name", type=str, default="baseline_all", help="Prefix for checkpoint names")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument(
        "--loss",
        type=str,
        default="infonce",
        choices=["infonce", "triplet", "batch_cls"],
        help="Loss function to use",
    )
    parser.add_argument("--features_dir", type=str, default="data/features", help="Directory containing feature tensors")
    parser.add_argument("--arch", type=str, default="baseline", choices=["baseline", "aacl"],
                        help="Mô hình: baseline=BaselineFusion (CLS+EOS), aacl=AACLFusion (patches+full text)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed cho model initialization và batch shuffle")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed cố định cho train/dev split")
    parser.add_argument("--dev_per_category", type=int, default=100,
                        help="Số dev samples lấy từ mỗi category")
    args = parser.parse_args()

    print("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (DAY 3) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Cấu hình (Config)
    config = {
        "run_name"    : args.run_name,
        "arch"        : args.arch,
        "batch_size"  : 256,
        "epochs"      : args.epochs,
        "lr"          : 1e-4,
        "weight_decay": 1e-4,
        "temperature" : 0.1,
        "dev_size"    : args.dev_per_category * 3,
        "dev_per_category": args.dev_per_category,
        "seed"        : args.seed,
        "split_seed"  : args.split_seed,
        "loss"        : args.loss,
        "features_dir": args.features_dir
    }
    if args.loss == "batch_cls":
        config.update({
            "loss_similarity" : "dot",
            "loss_normalize"  : False,
            "loss_temperature": None,
            "loss_symmetric"  : False
        })
    
    # Tạo thư mục checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_name = config["run_name"]
    with open(f"checkpoints/{ckpt_name}_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Cố định Seed
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    random.seed(config["seed"])
    
    # 2. Chuẩn bị Dữ liệu
    use_patches = (config["arch"] == "aacl")
    print(f"\n[1/4] Đang nạp Dataset (arch={config['arch']}, use_patches={use_patches})...")
    full_dataset = FashionIQDataset(
        data_dir="data",
        features_dir=config["features_dir"],
        category="all",
        use_patches=use_patches
    )
    
    train_subset, dev_subset, dev_counts = stratified_train_dev_split(
        full_dataset,
        dev_per_category=config["dev_per_category"],
        split_seed=config["split_seed"],
    )
    
    print(f"Tổng mẫu: {len(full_dataset)} | Train: {len(train_subset)} | Dev: {len(dev_subset)}")
    print(
        f"Dev split (split_seed={config['split_seed']}): "
        + ", ".join(
            f"{category}={count}"
            for category, count in sorted(dev_counts.items())
        )
    )
    
    collate_fn = aacl_collate_fn if use_patches else custom_collate_fn
    
    train_sampler = CategoryBatchSampler(train_subset, config["batch_size"], drop_last=True)
    train_loader  = DataLoader(train_subset, batch_sampler=train_sampler, collate_fn=collate_fn)
    dev_loader    = DataLoader(dev_subset, batch_size=config["batch_size"],
                               shuffle=False, drop_last=False, collate_fn=collate_fn)
    
    # 3. Khởi tạo Mô hình & Tối ưu hóa
    print(f"\n[2/4] Khởi tạo mô hình [{config['arch'].upper()}]...")
    if config["arch"] == "aacl":
        model = AACLFusion(img_dim=768, txt_dim=512, hidden_dim=768).to(device)
        print(
            "   Kiến trúc: AACLFusion "
            "(50 visual tokens = 1 CLS + 49 patches, full text sequence)"
        )
    else:
        model = BaselineFusion(img_dim=768, txt_dim=512, hidden_dim=1024, out_dim=768).to(device)
        print("   Kiến trúc: BaselineFusion (CLS token + EOS token, MLP)")
    if config["loss"] == "triplet":
        criterion = CIRTripletLoss(margin=0.2).to(device)
        print(f"   Hàm Loss: TripletMarginLoss (margin=0.2)")
    elif config["loss"] == "batch_cls":
        criterion = CIRBatchClassificationLoss(
            normalize=config["loss_normalize"],
            temperature=config["loss_temperature"],
            symmetric=config["loss_symmetric"]
        ).to(device)
        print("   Hàm Loss: AACL Batch Classification (one-way dot product)")
    else:
        criterion = CIRLoss(temperature=config["temperature"]).to(device)
        print(f"   Hàm Loss: InfoNCE (temperature={config['temperature']})")
    
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    # 4. Training Loop
    print(f"\n[3/4] BẮT ĐẦU HUẤN LUYỆN {config['epochs']} EPOCHS...")
    best_dev_acc = 0.0
    
    for epoch in range(1, config["epochs"] + 1):
        # --- TRAIN PHASE ---
        model.train()
        total_loss = 0.0
        total_query_norm = 0.0
        train_batch_count = 0
        
        for batch in train_loader:
            if use_patches:
                src_imgs, src_patches, txt_feats, txt_mask, txt_lengths, tgt_imgs = batch
                src_imgs    = src_imgs.to(device)
                src_patches = src_patches.to(device)
                txt_feats   = txt_feats.to(device)
                txt_mask    = txt_mask.to(device)
                tgt_imgs    = tgt_imgs.to(device)
            else:
                src_imgs, txt_feats, txt_mask, txt_lengths, tgt_imgs = batch
                src_imgs   = src_imgs.to(device)
                txt_feats  = txt_feats.to(device)
                txt_lengths= txt_lengths.to(device)
                tgt_imgs   = tgt_imgs.to(device)
            
            optimizer.zero_grad()
            
            if use_patches:
                combined_query = model(src_patches, txt_feats, txt_mask)
            else:
                B           = src_imgs.size(0)
                eos_indices = txt_lengths - 1
                txt_eos     = txt_feats[torch.arange(B, device=device), eos_indices]
                combined_query = model(src_imgs, txt_eos)
            loss = criterion(combined_query, tgt_imgs)

            total_query_norm += (
                combined_query.detach().norm(dim=-1).mean().item()
            )
            train_batch_count += 1

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_query_norm = total_query_norm / train_batch_count
        
        # Step LR Scheduler
        scheduler.step()
        
        # --- EVAL PHASE (Dev Subset) ---
        dev_acc = eval_accuracy(model, dev_loader, device, arch=config["arch"])
        
        print(
            f"Epoch [{epoch:02d}/{config['epochs']}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Mean Query Norm: {avg_query_norm:.4f} | "
            f"Dev Acc@1: {dev_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # --- CHECKPOINTING ---
        # 1. Luôn lưu last
        torch.save(model.state_dict(), f"checkpoints/{ckpt_name}_last.pth")
        
        # 2. Lưu best nếu Dev Acc tăng
        if dev_acc > best_dev_acc:
            print(f"   >>> Dev Acc tăng từ {best_dev_acc:.2f}% lên {dev_acc:.2f}%. Lưu model tốt nhất tại checkpoints/{ckpt_name}_best.pth (Accuracy@1: {dev_acc:.2f}%)")
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), f"checkpoints/{ckpt_name}_best.pth")

    print(f"\n[4/4] HUẤN LUYỆN HOÀN TẤT! Best Dev Accuracy: {best_dev_acc:.2f}%")
    print("Checkpoints đã được lưu tại thư mục 'checkpoints/'")

if __name__ == "__main__":
    main()
