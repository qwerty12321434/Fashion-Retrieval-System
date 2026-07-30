import os
import sys
import io
import json
import torch

# Ép Windows Terminal dùng UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    features_dir = "data/features_fashionclip"
    data_dir     = "data"

    print("=" * 60)
    print("  LỌC CANDIDATE PATCH TOKENS")
    print(f"  Input : {features_dir}/all_image_tokens.pt (~11GB)")
    print(f"  Output: {features_dir}/candidate_patch_tokens.pt (~1-2GB)")
    print("=" * 60)

    # 1. Gom toàn bộ candidate ASIN từ train + val JSON
    print("\n[1/3] Thu thập candidate ASINs từ tất cả file JSON...")
    candidate_asins = set()
    splits = ["train", "val"]
    categories = ["dress", "shirt", "toptee"]

    for cat in categories:
        for split in splits:
            json_path = os.path.join(data_dir, f"json/cap.{cat}.{split}.json")
            if not os.path.exists(json_path):
                print(f"  [BỎ QUA] {json_path} không tồn tại")
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            before = len(candidate_asins)
            for item in data:
                candidate_asins.add(item["candidate"])
            print(f"  {json_path}: +{len(candidate_asins) - before} ASINs mới (tổng: {len(candidate_asins)})")

    print(f"\n  Tổng candidate ASIN cần lọc: {len(candidate_asins)}")

    # 2. Nạp toàn bộ image tokens (RAM-heavy step)
    all_tokens_path = os.path.join(features_dir, "all_image_tokens.pt")
    if not os.path.exists(all_tokens_path):
        print(f"\n[LỖI] Không tìm thấy: {all_tokens_path}")
        print("  Cần chạy lại: python scripts/extract_features.py --backbone fashionclip")
        return

    print(f"\n[2/3] Đang nạp {all_tokens_path} vào RAM (có thể mất 1-2 phút)...")
    all_tokens = torch.load(all_tokens_path, weights_only=True, mmap=True)
    print(f"  Đã nạp {len(all_tokens)} ASIN, mỗi ASIN shape: {next(iter(all_tokens.values())).shape}")

    # 3. Lọc và lưu
    print(f"\n[3/3] Lọc ra {len(candidate_asins)} candidate ASINs...")
    filtered = {}
    missing  = []
    for asin in candidate_asins:
        if asin in all_tokens:
            # Clone để tách khỏi storage batch lớn của all_image_tokens.pt.
            # Nếu giữ tensor view, torch.save sẽ ghi gần như toàn bộ file 11GB.
            filtered[asin] = all_tokens[asin].clone()  # tensor [50, 768]
        else:
            missing.append(asin)

    print(f"  Tìm thấy: {len(filtered)} / {len(candidate_asins)} ASINs")
    if missing:
        print(f"  [CẢNH BÁO] {len(missing)} ASINs không có trong all_image_tokens.pt:")
        for a in missing[:5]:
            print(f"    - {a}")
        if len(missing) > 5:
            print(f"    ... (còn {len(missing) - 5} nữa)")

    out_path = os.path.join(features_dir, "candidate_patch_tokens.pt")
    temp_path = out_path + ".tmp"
    torch.save(filtered, temp_path)
    os.replace(temp_path, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\n  [OK] Đã lưu {len(filtered)} candidate patch tensors -> {out_path}")
    print(f"  Kích thước file: {size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("  HOÀN TẤT! Giờ có thể train AACLFusion:")
    print("  python src/train.py --arch aacl --run_name aacl_fashionclip_triplet \\")
    print("    --loss triplet --features_dir data/features_fashionclip")
    print("=" * 60)

if __name__ == "__main__":
    main()
