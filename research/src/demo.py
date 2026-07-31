import argparse
import io
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor

from models.model import AACLFusion, BaselineFusion


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def image_path_for_asin(image_dir, asin):
    for extension in (".jpg", ".png"):
        path = Path(image_dir) / f"{asin}{extension}"
        if path.exists():
            return path
    return Path(image_dir) / f"{asin}.jpg"


def load_font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def target_rank(similarities, target_idx):
    if target_idx is None:
        return None
    return 1 + int(
        (similarities > similarities[target_idx]).sum().item()
    )


def load_fusion_model(arch, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if arch == "aacl":
        model = AACLFusion(img_dim=768, txt_dim=512, hidden_dim=768).to(device)
    else:
        hidden_dim = state_dict["mlp.0.weight"].shape[0]
        model = BaselineFusion(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_validation_query(args, gallery_asins, gallery_cls, gallery_embeds, device):
    json_path = Path("data/json") / f"cap.{args.category}.val.json"
    with open(json_path, encoding="utf-8") as handle:
        items = json.load(handle)
    if args.val_index < 0 or args.val_index >= len(items):
        raise ValueError(
            f"--val_index phải nằm trong [0, {len(items) - 1}] cho category {args.category}"
        )

    item = items[args.val_index]
    candidate = item["candidate"]
    target = item["target"]
    modifier = " and ".join(item["captions"])
    asin_to_idx = {asin: idx for idx, asin in enumerate(gallery_asins)}
    if candidate not in asin_to_idx or target not in asin_to_idx:
        raise KeyError("Candidate hoặc target của validation query không có trong gallery")

    hidden = torch.load(
        Path(args.features_dir) / f"{args.category}_val_text_hidden.pt",
        map_location="cpu",
        weights_only=True,
    )
    embeds = torch.load(
        Path(args.features_dir) / f"{args.category}_val_text_embeds.pt",
        map_location="cpu",
        weights_only=True,
    )

    candidate_idx = asin_to_idx[candidate]
    candidate_cls = gallery_cls[candidate_idx].unsqueeze(0)
    candidate_embed = gallery_embeds[candidate_idx].unsqueeze(0)
    text_hidden = hidden[args.val_index].unsqueeze(0).to(device)
    text_embed = embeds[args.val_index].unsqueeze(0).to(device)
    text_mask = torch.ones(
        1, text_hidden.size(1), dtype=torch.bool, device=device
    )

    candidate_tokens = None
    if args.arch == "aacl":
        patch_path = Path(args.features_dir) / "candidate_patch_tokens.pt"
        patches = torch.load(
            patch_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if candidate not in patches:
            raise KeyError(f"Không có patch token cho candidate {candidate}")
        candidate_tokens = patches[candidate].unsqueeze(0).to(device)

    return {
        "mode": f"validation/{args.category}[{args.val_index}]",
        "candidate": candidate,
        "target": target,
        "text": modifier,
        "candidate_cls": candidate_cls,
        "candidate_embed": candidate_embed,
        "candidate_tokens": candidate_tokens,
        "text_hidden": text_hidden,
        "text_embed": text_embed,
        "text_mask": text_mask,
        "text_eos": text_hidden[:, -1, :],
    }


def load_free_form_query(args, device):
    if not args.candidate or not args.text:
        raise ValueError(
            "Free-form mode cần cả --candidate và --text. "
            "Hoặc dùng --val_index để chạy validation offline."
        )

    candidate_path = image_path_for_asin(args.image_dir, args.candidate)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh candidate: {candidate_path}")

    print(f"1. Nạp backbone {args.backbone} cho free-form mode...")
    try:
        processor = CLIPProcessor.from_pretrained(
            args.backbone,
            local_files_only=args.local_files_only,
        )
        clip_model = CLIPModel.from_pretrained(
            args.backbone,
            use_safetensors=True,
            local_files_only=args.local_files_only,
        ).to(device)
    except Exception as error:
        raise RuntimeError(
            "Không nạp được backbone/processor. Nếu máy không có mạng hoặc model chưa "
            "được cache đầy đủ, hãy dùng validation offline với --val_index."
        ) from error
    clip_model.eval()

    image = Image.open(candidate_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(
        text=args.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        image_outputs = clip_model.vision_model(**image_inputs)
        candidate_tokens = image_outputs.last_hidden_state
        candidate_cls = candidate_tokens[:, 0, :]
        candidate_embed = clip_model.visual_projection(
            clip_model.vision_model.post_layernorm(candidate_cls)
        )

        text_outputs = clip_model.text_model(**text_inputs)
        text_hidden = text_outputs.last_hidden_state
        eos_indices = text_inputs.input_ids.to(torch.int).argmax(dim=-1)
        text_eos = text_hidden[
            torch.arange(text_hidden.shape[0], device=device),
            eos_indices,
        ]
        text_embed = clip_model.text_projection(text_eos)

    return {
        "mode": "free-form",
        "candidate": args.candidate,
        "target": args.target,
        "text": args.text,
        "candidate_cls": candidate_cls,
        "candidate_embed": candidate_embed,
        "candidate_tokens": candidate_tokens,
        "text_hidden": text_hidden,
        "text_embed": text_embed,
        "text_mask": text_inputs.attention_mask.bool().to(device),
        "text_eos": text_eos,
    }


def retrieve(query, gallery, gallery_asins, top_k, target_idx):
    query_norm = F.normalize(query, p=2, dim=-1)
    gallery_norm = F.normalize(gallery, p=2, dim=-1)
    similarities = (query_norm @ gallery_norm.T).squeeze(0)
    scores, indices = torch.topk(similarities, k=top_k)
    results = [
        (gallery_asins[int(index)], float(score))
        for index, score in zip(indices, scores)
    ]
    return results, target_rank(similarities, target_idx)


def draw_demo(args, query, zero_results, fusion_results, zero_rank, fusion_rank):
    top_k = args.top_k
    cell_w, cell_h, header_h = 250, 315, 145
    cols, rows = top_k + 1, 2
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    label_font = load_font(17, bold=True)
    text_font = load_font(15)
    small_font = load_font(13)

    model_label = (
        "AACL-INSPIRED FUSION" if args.arch == "aacl" else "BASELINE FUSION"
    )
    title = f"CIR DEMO — ZERO-SHOT vs {model_label}"
    draw.text((18, 12), title, fill="#111827", font=title_font)

    wrapped_text = textwrap.wrap(f"Modifier: {query['text']}", width=max(75, cols * 20))
    for line_index, line in enumerate(wrapped_text[:2]):
        draw.text(
            (18, 47 + line_index * 21),
            line,
            fill="#1D4ED8",
            font=text_font,
        )

    target_text = query["target"] or "không cung cấp"
    rank_text = (
        f"Rank zero-shot: {zero_rank if zero_rank is not None else 'N/A'}  |  "
        f"Rank fusion: {fusion_rank if fusion_rank is not None else 'N/A'}"
    )
    draw.text(
        (18, 96),
        (
            f"Mode: {query['mode']}  |  Candidate: {query['candidate']}  |  "
            f"Target: {target_text}  |  {rank_text}"
        ),
        fill="#374151",
        font=small_font,
    )
    draw.text(
        (18, 119),
        (
            f"Checkpoint: {args.ckpt}  |  "
            f"Gallery: {args.gallery_label} ({len(args.gallery_asins):,} ảnh)"
        ),
        fill="#6B7280",
        font=small_font,
    )

    def draw_product(col, row, asin, label, score=None, role=None):
        x = col * cell_w
        y = header_h + row * cell_h
        is_target = query["target"] is not None and asin == query["target"]
        if is_target:
            color = "#047857"
        elif role == "candidate":
            color = "#1D4ED8"
        else:
            color = "#111827"

        draw.text((x + 10, y + 8), label, fill=color, font=label_font)
        info = f"ASIN: {asin}"
        if score is not None:
            info += f" | sim={score:.3f}"
        draw.text((x + 10, y + 35), info, fill=color, font=text_font)

        path = image_path_for_asin(args.image_dir, asin)
        if path.exists():
            try:
                image = Image.open(path).convert("RGB")
                image.thumbnail((cell_w - 24, cell_h - 75))
                px = x + (cell_w - image.width) // 2
                py = y + 66 + (cell_h - 75 - image.height) // 2
                canvas.paste(image, (px, py))
            except Exception:
                draw.text(
                    (x + 20, y + 100),
                    "Lỗi đọc ảnh",
                    fill="red",
                    font=text_font,
                )
        else:
            draw.text(
                (x + 20, y + 100),
                "Không tìm thấy ảnh",
                fill="red",
                font=text_font,
            )

        if is_target:
            draw.rectangle(
                (x + 3, y + 3, x + cell_w - 3, y + cell_h - 3),
                outline="#10B981",
                width=5,
            )
        elif role == "candidate":
            draw.rectangle(
                (x + 3, y + 3, x + cell_w - 3, y + cell_h - 3),
                outline="#3B82F6",
                width=4,
            )

    draw_product(
        0,
        0,
        query["candidate"],
        "Ảnh tham chiếu",
        role="candidate",
    )
    if query["target"]:
        draw_product(0, 1, query["target"], "Ảnh đích thật", role="target")
    else:
        x, y = 0, header_h + cell_h
        draw.text((12, y + 12), "Target chưa cung cấp", fill="#6B7280", font=label_font)
        for line_index, line in enumerate(
            textwrap.wrap(
                "Dùng --target hoặc --val_index để hiển thị target và rank thật.",
                width=27,
            )
        ):
            draw.text(
                (12, y + 52 + line_index * 20),
                line,
                fill="#6B7280",
                font=text_font,
            )

    for col, (asin, score) in enumerate(zero_results, start=1):
        draw_product(col, 0, asin, f"Zero-shot Top-{col}", score)
    for col, (asin, score) in enumerate(fusion_results, start=1):
        draw_product(col, 1, asin, f"{args.arch.upper()} Top-{col}", score)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"\n=> Đã lưu demo: {output_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Demo CIR: free-form query hoặc FashionIQ validation offline. "
            "So sánh zero-shot với một fusion checkpoint."
        )
    )
    parser.add_argument("--candidate", help="ASIN ảnh tham chiếu cho free-form mode")
    parser.add_argument("--text", help="Modifier text cho free-form mode")
    parser.add_argument(
        "--target",
        help="ASIN target tùy chọn; dùng để highlight và tính full-gallery rank",
    )
    parser.add_argument(
        "--val_index",
        type=int,
        help="Validation index; bật offline mode và tự lấy candidate/text/target",
    )
    parser.add_argument(
        "--category",
        default="dress",
        choices=["dress", "shirt", "toptee"],
        help="Category dùng cùng --val_index",
    )
    parser.add_argument(
        "--gallery_scope",
        default="global",
        choices=["global", "category"],
        help=(
            "global: tìm trên 74K ảnh; category: tìm trên split val "
            "của --category (chỉ dùng với --val_index)"
        ),
    )
    parser.add_argument(
        "--splits_dir",
        default="data/json",
        help="Thư mục chứa split.{category}.val.json",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Số kết quả hiển thị cho mỗi hàng (1-10)",
    )
    parser.add_argument(
        "--output",
        default="demo_result.png",
        help="Đường dẫn ảnh output",
    )
    parser.add_argument(
        "--ckpt",
        default="fashionclip_triplet_1024_best.pth",
        help="Checkpoint fusion trong thư mục checkpoints/",
    )
    parser.add_argument(
        "--backbone",
        default="patrickjohncyh/fashion-clip",
        help="Hugging Face backbone cho free-form mode",
    )
    parser.add_argument(
        "--features_dir",
        default="data/features_fashionclip",
        help="Thư mục gallery/text features",
    )
    parser.add_argument(
        "--image_dir",
        default=r"E:\MyDownloads\fashion-iq-dataset\fashionIQ_dataset\images",
        help="Thư mục ảnh FashionIQ",
    )
    parser.add_argument(
        "--arch",
        default="baseline",
        choices=["baseline", "aacl"],
        help="Kiến trúc checkpoint",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Không truy cập mạng khi nạp backbone ở free-form mode",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 10:
        parser.error("--top_k phải nằm trong [1, 10]")
    if args.val_index is None and (not args.candidate or not args.text):
        parser.error("Cần --candidate và --text, hoặc dùng --val_index")
    if args.gallery_scope == "category" and args.val_index is None:
        parser.error("--gallery_scope category chỉ dùng cùng --val_index")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    checkpoint_path = Path("checkpoints") / args.ckpt
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    print(f"[INFO] Checkpoint: {checkpoint_path.resolve()}")
    print(f"[INFO] Modified: {time.ctime(checkpoint_path.stat().st_mtime)}")

    feature_dir = Path(args.features_dir)
    gallery_cls = torch.load(
        feature_dir / "gallery_cls_768.pt",
        map_location=device,
        weights_only=True,
    )
    gallery_embeds = torch.load(
        feature_dir / "gallery_embeds_512.pt",
        map_location=device,
        weights_only=True,
    )
    with open(feature_dir / "gallery_asins.json", encoding="utf-8") as handle:
        gallery_asins = json.load(handle)
    global_gallery_asins = gallery_asins

    fusion_model = load_fusion_model(
        args.arch,
        checkpoint_path,
        device,
    )
    if args.val_index is not None:
        print(
            f"1. Validation offline: category={args.category}, index={args.val_index}"
        )
        query = load_validation_query(
            args,
            global_gallery_asins,
            gallery_cls,
            gallery_embeds,
            device,
        )
    else:
        query = load_free_form_query(args, device)

    if args.gallery_scope == "category":
        split_path = (
            Path(args.splits_dir) / f"split.{args.category}.val.json"
        )
        if not split_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy FashionIQ split: {split_path}"
            )
        with open(split_path, encoding="utf-8") as handle:
            split_asins = json.load(handle)
        if len(split_asins) != len(set(split_asins)):
            raise ValueError(f"{split_path} chứa ASIN trùng lặp")

        global_asin_to_idx = {
            asin: index
            for index, asin in enumerate(global_gallery_asins)
        }
        missing = [
            asin for asin in split_asins if asin not in global_asin_to_idx
        ]
        if missing:
            raise ValueError(
                f"{split_path}: thiếu {len(missing)} ảnh trong gallery local"
            )
        gallery_indices = torch.tensor(
            [global_asin_to_idx[asin] for asin in split_asins],
            dtype=torch.long,
            device=device,
        )
        gallery_asins = split_asins
        gallery_cls = gallery_cls.index_select(0, gallery_indices)
        gallery_embeds = gallery_embeds.index_select(0, gallery_indices)
        args.gallery_label = f"FashionIQ {args.category}.val"
    else:
        gallery_asins = global_gallery_asins
        args.gallery_label = "Global-74K"

    args.gallery_asins = gallery_asins
    asin_to_idx = {
        asin: index for index, asin in enumerate(gallery_asins)
    }
    print(
        f"[INFO] Gallery scope: {args.gallery_label}, "
        f"{len(gallery_asins):,} ảnh"
    )

    target_idx = None
    if query["target"]:
        if query["target"] not in asin_to_idx:
            raise KeyError(f"Target {query['target']} không có trong gallery")
        target_idx = asin_to_idx[query["target"]]

    print("2. Tính zero-shot vector addition...")
    zero_query = F.normalize(
        F.normalize(query["candidate_embed"], dim=-1)
        + F.normalize(query["text_embed"], dim=-1),
        dim=-1,
    )
    zero_results, zero_rank = retrieve(
        zero_query,
        gallery_embeds,
        gallery_asins,
        args.top_k,
        target_idx,
    )

    print(f"3. Tính {args.arch.upper()} fusion...")
    with torch.no_grad():
        if args.arch == "aacl":
            fusion_query = fusion_model(
                query["candidate_tokens"],
                query["text_hidden"],
                query["text_mask"],
            )
        else:
            fusion_query = fusion_model(
                query["candidate_cls"],
                query["text_eos"],
            )
    fusion_results, fusion_rank = retrieve(
        fusion_query,
        gallery_cls,
        gallery_asins,
        args.top_k,
        target_idx,
    )

    print("4. Tạo ảnh trực quan...")
    draw_demo(
        args,
        query,
        zero_results,
        fusion_results,
        zero_rank,
        fusion_rank,
    )

    print("\n=== TÓM TẮT ===")
    print(f"Mode      : {query['mode']}")
    print(f"Candidate : {query['candidate']}")
    print(f"Modifier  : {query['text']}")
    print(f"Target    : {query['target'] or 'N/A'}")
    print(f"Zero rank : {zero_rank if zero_rank is not None else 'N/A'}")
    print(f"Fusion rank: {fusion_rank if fusion_rank is not None else 'N/A'}")
    print("Zero-shot :", [asin for asin, _ in zero_results])
    print("Fusion    :", [asin for asin, _ in fusion_results])


if __name__ == "__main__":
    main()
