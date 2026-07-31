import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.dataset import validate_patch_features
from models.model import AACLFusion, BaselineFusion


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CATEGORIES = ("dress", "shirt", "toptee")


def get_rank(similarities, target_idx):
    """Return the official-style one-based rank using strictly better scores."""
    target_score = similarities[target_idx]
    return 1 + int((similarities > target_score).sum().item())


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_gallery_view(
    gallery_asins,
    gallery_cls,
    gallery_embeds,
    split_asins=None,
    context="global",
):
    """Build a global or split-specific gallery with local ASIN indices."""
    if split_asins is None:
        selected_asins = list(gallery_asins)
        indices = None
    else:
        selected_asins = list(split_asins)
        if len(selected_asins) != len(set(selected_asins)):
            raise ValueError(f"{context}: split chứa ASIN trùng lặp")

        global_asin_to_idx = {
            asin: index for index, asin in enumerate(gallery_asins)
        }
        missing = [
            asin for asin in selected_asins if asin not in global_asin_to_idx
        ]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"{context}: thiếu {len(missing)} ảnh split trong gallery. "
                f"Ví dụ: {preview}"
            )
        indices = torch.tensor(
            [global_asin_to_idx[asin] for asin in selected_asins],
            dtype=torch.long,
            device=gallery_cls.device,
        )

    if indices is None:
        selected_cls = gallery_cls
        selected_embeds = gallery_embeds
    else:
        selected_cls = gallery_cls.index_select(0, indices)
        selected_embeds = gallery_embeds.index_select(0, indices)

    return {
        "asins": selected_asins,
        "asin_to_idx": {
            asin: index for index, asin in enumerate(selected_asins)
        },
        "cls_norm": F.normalize(selected_cls, p=2, dim=-1),
        "embeds_norm": F.normalize(selected_embeds, p=2, dim=-1),
    }


def macro_average(category_metrics, metric_name):
    return sum(
        category_metrics[category][metric_name] for category in CATEGORIES
    ) / len(CATEGORIES)


def load_model(arch, checkpoint_path, device):
    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if arch == "aacl":
        model = AACLFusion(
            img_dim=768, txt_dim=512, hidden_dim=768
        ).to(device)
    else:
        hidden_dim = state_dict["mlp.0.weight"].shape[0]
        model = BaselineFusion(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_category(
    category,
    annotations,
    text_hidden,
    text_embeds,
    gallery_view,
    gallery_cls,
    gallery_embeds,
    global_asin_to_idx,
    models,
    arch,
    candidate_patch_tokens,
    device,
):
    method_names = ["zero_shot"] + list(models)
    hits = {
        method_name: {"r10_hits": 0, "r50_hits": 0}
        for method_name in method_names
    }
    valid_queries = 0

    for index, item in enumerate(
        tqdm(annotations, desc=f"{category}/{len(gallery_view['asins'])}")
    ):
        candidate_asin = item["candidate"]
        target_asin = item["target"]

        if candidate_asin not in global_asin_to_idx:
            raise ValueError(
                f"{category}[{index}]: candidate {candidate_asin} "
                "không có trong global gallery"
            )
        if target_asin not in gallery_view["asin_to_idx"]:
            raise ValueError(
                f"{category}[{index}]: target {target_asin} "
                "không có trong gallery của protocol"
            )

        candidate_global_idx = global_asin_to_idx[candidate_asin]
        target_local_idx = gallery_view["asin_to_idx"][target_asin]
        valid_queries += 1

        candidate_embed = F.normalize(
            gallery_embeds[candidate_global_idx].unsqueeze(0),
            p=2,
            dim=-1,
        )
        current_text_embed = F.normalize(
            text_embeds[index].unsqueeze(0), p=2, dim=-1
        )
        zero_query = F.normalize(
            candidate_embed + current_text_embed, p=2, dim=-1
        )
        zero_similarities = (
            zero_query @ gallery_view["embeds_norm"].T
        ).squeeze(0)
        zero_rank = get_rank(zero_similarities, target_local_idx)
        if zero_rank <= 10:
            hits["zero_shot"]["r10_hits"] += 1
        if zero_rank <= 50:
            hits["zero_shot"]["r50_hits"] += 1

        candidate_cls = gallery_cls[candidate_global_idx].unsqueeze(0)
        current_text_hidden = text_hidden[index].unsqueeze(0)

        for model_name, model in models.items():
            if arch == "aacl":
                patches = candidate_patch_tokens[candidate_asin]
                patches = patches.unsqueeze(0).to(device)
                text_mask = torch.ones(
                    1,
                    current_text_hidden.size(1),
                    dtype=torch.bool,
                    device=device,
                )
                fusion_query = model(
                    patches, current_text_hidden, text_mask
                )
            else:
                text_eos = current_text_hidden[:, -1, :]
                fusion_query = model(candidate_cls, text_eos)

            fusion_query = F.normalize(fusion_query, p=2, dim=-1)
            similarities = (
                fusion_query @ gallery_view["cls_norm"].T
            ).squeeze(0)
            rank = get_rank(similarities, target_local_idx)
            if rank <= 10:
                hits[model_name]["r10_hits"] += 1
            if rank <= 50:
                hits[model_name]["r50_hits"] += 1

    if valid_queries == 0:
        raise ValueError(f"{category}: không có validation query hợp lệ")

    return {
        method_name: {
            "queries": valid_queries,
            "gallery_size": len(gallery_view["asins"]),
            "r10_hits": values["r10_hits"],
            "r50_hits": values["r50_hits"],
            "r10": values["r10_hits"] / valid_queries * 100,
            "r50": values["r50_hits"] / valid_queries * 100,
        }
        for method_name, values in hits.items()
    }


def aggregate_results(results_by_category, method_name):
    category_metrics = {
        category: results_by_category[category][method_name]
        for category in CATEGORIES
    }
    total_queries = sum(
        metrics["queries"] for metrics in category_metrics.values()
    )
    total_r10_hits = sum(
        metrics["r10_hits"] for metrics in category_metrics.values()
    )
    total_r50_hits = sum(
        metrics["r50_hits"] for metrics in category_metrics.values()
    )
    return {
        "categories": category_metrics,
        "macro_average": {
            "r10": macro_average(category_metrics, "r10"),
            "r50": macro_average(category_metrics, "r50"),
        },
        "pooled": {
            "queries": total_queries,
            "r10": total_r10_hits / total_queries * 100,
            "r50": total_r50_hits / total_queries * 100,
        },
    }


def print_method_result(label, result, protocol):
    print(f"\n{label}")
    for category in CATEGORIES:
        metrics = result["categories"][category]
        print(
            f" - {category:7s} | gallery={metrics['gallery_size']:5d} "
            f"| queries={metrics['queries']:4d} "
            f"| R@10={metrics['r10']:6.2f}% "
            f"| R@50={metrics['r50']:6.2f}%"
        )

    macro = result["macro_average"]
    pooled = result["pooled"]
    print(
        f" - Macro average       | R@10={macro['r10']:.2f}% "
        f"| R@50={macro['r50']:.2f}%"
    )
    if protocol == "global":
        print(
            f" - Global pooled       | queries={pooled['queries']} "
            f"| R@10={pooled['r10']:.2f}% "
            f"| R@50={pooled['r50']:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CIR models with FashionIQ or global protocol"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="Một hoặc nhiều checkpoint trong checkpoints/",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features",
        help="Thư mục feature; dùng data/features_fashionclip cho FashionCLIP",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="baseline",
        choices=["baseline", "aacl"],
    )
    parser.add_argument(
        "--protocol",
        choices=["fashioniq", "global"],
        default="fashioniq",
        help="fashioniq: gallery val theo category; global: toàn bộ catalog local",
    )
    parser.add_argument(
        "--splits_dir",
        default="data/json",
        help="Thư mục chứa cap.*.val.json và split.*.val.json",
    )
    parser.add_argument(
        "--output_json",
        help="Đường dẫn tùy chọn để lưu kết quả JSON có cấu trúc",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_dir = Path(args.features_dir)
    splits_dir = Path(args.splits_dir)

    print("=== ĐÁNH GIÁ COMPOSED IMAGE RETRIEVAL ===")
    print(f"Thiết bị : {device}")
    print(f"Features : {features_dir.resolve()}")
    print(
        "Protocol : "
        + (
            "FashionIQ Standard (category-specific val gallery)"
            if args.protocol == "fashioniq"
            else "Global-74K (toàn bộ catalog local)"
        )
    )

    gallery_cls = torch.load(
        features_dir / "gallery_cls_768.pt",
        map_location=device,
        weights_only=True,
    )
    gallery_embeds = torch.load(
        features_dir / "gallery_embeds_512.pt",
        map_location=device,
        weights_only=True,
    )
    gallery_asins = load_json(features_dir / "gallery_asins.json")
    if len(gallery_asins) != gallery_cls.shape[0]:
        raise ValueError("gallery_asins và gallery_cls không cùng số phần tử")
    if len(gallery_asins) != gallery_embeds.shape[0]:
        raise ValueError("gallery_asins và gallery_embeds không cùng số phần tử")
    global_asin_to_idx = {
        asin: index for index, asin in enumerate(gallery_asins)
    }
    print(f"Global gallery: {len(gallery_asins):,} ảnh")

    category_data = {}
    all_candidates = []
    for category in CATEGORIES:
        annotations = load_json(
            splits_dir / f"cap.{category}.val.json"
        )
        text_hidden = torch.load(
            features_dir / f"{category}_val_text_hidden.pt",
            map_location=device,
            weights_only=True,
        )
        text_embeds = torch.load(
            features_dir / f"{category}_val_text_embeds.pt",
            map_location=device,
            weights_only=True,
        )
        if len(annotations) != len(text_hidden):
            raise ValueError(
                f"{category}: annotation ({len(annotations)}) và "
                f"text hidden ({len(text_hidden)}) không khớp"
            )
        if len(annotations) != len(text_embeds):
            raise ValueError(
                f"{category}: annotation ({len(annotations)}) và "
                f"text embeds ({len(text_embeds)}) không khớp"
            )

        if args.protocol == "fashioniq":
            split_path = splits_dir / f"split.{category}.val.json"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Thiếu FashionIQ split chính thức: {split_path}"
                )
            split_asins = load_json(split_path)
        else:
            split_asins = None

        gallery_view = build_gallery_view(
            gallery_asins,
            gallery_cls,
            gallery_embeds,
            split_asins=split_asins,
            context=f"{category}.val",
        )
        category_data[category] = {
            "annotations": annotations,
            "text_hidden": text_hidden,
            "text_embeds": text_embeds,
            "gallery_view": gallery_view,
        }
        all_candidates.extend(
            item["candidate"] for item in annotations
        )
        print(
            f"{category:7s}: queries={len(annotations):,}, "
            f"gallery={len(gallery_view['asins']):,}"
        )

    candidate_patch_tokens = None
    if args.arch == "aacl":
        patch_path = features_dir / "candidate_patch_tokens.pt"
        if not patch_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {patch_path}. "
                "Chạy scripts/prep_candidate_patches.py trước."
            )
        print(f"Nạp candidate patch tokens: {patch_path}")
        candidate_patch_tokens = torch.load(
            patch_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        validate_patch_features(
            candidate_patch_tokens,
            all_candidates,
            expected_shape=(50, gallery_cls.shape[-1]),
            context="validation",
        )

    models = {}
    checkpoint_metadata = {}
    for checkpoint_name in args.ckpt:
        checkpoint_path = Path("checkpoints") / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Thiếu checkpoint: {checkpoint_path}")
        run_name = (
            checkpoint_name.replace("_best.pth", "")
            .replace("_last.pth", "")
            .replace(".pth", "")
        )
        print(
            f"Nạp checkpoint: {checkpoint_name} "
            f"(modified: {time.ctime(os.path.getmtime(checkpoint_path))})"
        )
        models[run_name] = load_model(
            args.arch, checkpoint_path, device
        )
        checkpoint_metadata[run_name] = checkpoint_name

    results_by_category = {}
    with torch.no_grad():
        for category in CATEGORIES:
            inputs = category_data[category]
            results_by_category[category] = evaluate_category(
                category=category,
                annotations=inputs["annotations"],
                text_hidden=inputs["text_hidden"],
                text_embeds=inputs["text_embeds"],
                gallery_view=inputs["gallery_view"],
                gallery_cls=gallery_cls,
                gallery_embeds=gallery_embeds,
                global_asin_to_idx=global_asin_to_idx,
                models=models,
                arch=args.arch,
                candidate_patch_tokens=candidate_patch_tokens,
                device=device,
            )

    output = {
        "protocol": args.protocol,
        "protocol_label": (
            "FashionIQ Standard"
            if args.protocol == "fashioniq"
            else "Global-74K"
        ),
        "architecture": args.arch,
        "features_dir": str(features_dir),
        "global_gallery_size": len(gallery_asins),
        "methods": {},
    }

    print("\n=== KẾT QUẢ ===")
    zero_result = aggregate_results(
        results_by_category, "zero_shot"
    )
    output["methods"]["zero_shot"] = {
        "checkpoint": None,
        **zero_result,
    }
    print_method_result("ZERO-SHOT VECTOR ADDITION", zero_result, args.protocol)

    for run_name in models:
        result = aggregate_results(results_by_category, run_name)
        output["methods"][run_name] = {
            "checkpoint": checkpoint_metadata[run_name],
            **result,
        }
        label = (
            "AACL FUSION" if args.arch == "aacl" else "BASELINE FUSION"
        )
        print_method_result(
            f"{label} ({run_name})", result, args.protocol
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu kết quả: {output_path.resolve()}")


if __name__ == "__main__":
    main()
