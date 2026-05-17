#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["row"] = int(row["row"])
        row["query_original_idx"] = int(row["query_original_idx"])
        row["source_original_idx"] = int(row["source_original_idx"])
        row["self_match"] = str(row["self_match"]).lower() == "true"
    return rows


def infer_cell_size(width, nrow, padding):
    usable = width - padding * (nrow + 1)
    if usable <= 0 or usable % nrow != 0:
        raise ValueError(f"Cannot infer cell size from width={width}, nrow={nrow}, padding={padding}")
    return usable // nrow


def crop_cell(grid, row, col, cell_size, padding):
    left = padding + col * (cell_size + padding)
    top = padding + row * (cell_size + padding)
    return np.array(grid.crop((left, top, left + cell_size, top + cell_size)).convert("RGB"))


def compute_ssim(gt_uint8, pred_uint8):
    return float(structural_similarity(gt_uint8, pred_uint8, data_range=255, channel_axis=-1))


def compute_mse(gt_uint8, pred_uint8):
    gt = gt_uint8.astype(np.float32) / 255.0
    pred = pred_uint8.astype(np.float32) / 255.0
    return float(np.square(gt - pred).mean())


class ClipImageSimilarity:
    def __init__(self, model_path, device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, gt_images, pred_images, batch_size):
        scores = []
        for start in tqdm(range(0, len(gt_images), batch_size), desc="CLIP"):
            end = min(start + batch_size, len(gt_images))
            images = []
            for gt, pred in zip(gt_images[start:end], pred_images[start:end]):
                images.append(Image.fromarray(gt))
                images.append(Image.fromarray(pred))
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1)
            gt_features = features[0::2]
            pred_features = features[1::2]
            batch_scores = (gt_features * pred_features).sum(dim=-1)
            scores.extend(float(x) for x in batch_scores.detach().cpu())
        return scores


def summarize(records):
    summary = {}
    groups = {
        "all": records,
        "self_match": [r for r in records if r["self_match"]],
        "non_self_match": [r for r in records if not r["self_match"]],
    }
    variants = sorted({r["variant"] for r in records})
    metrics = ["ssim", "mse", "clip_similarity"]
    for group_name, group_records in groups.items():
        summary[group_name] = {}
        for variant in variants:
            subset = [r for r in group_records if r["variant"] == variant]
            summary[group_name][variant] = {"count": len(subset)}
            for metric in metrics:
                values = np.array([r[metric] for r in subset], dtype=np.float64)
                if len(values) == 0:
                    summary[group_name][variant][metric] = None
                    continue
                summary[group_name][variant][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_path", type=str, required=True)
    parser.add_argument("--summary_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--clip_model_path", type=str,
                        default="/home/yiqiuliu/DreamDiffusion_old/pretrains/models/eeg_pretrain_scp/clip_vit_large_patch14")
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_clip", type=parse_bool, default=True)
    args = parser.parse_args()

    grid_path = Path(args.grid_path)
    summary_path = Path(args.summary_path)
    output_dir = Path(args.output_dir) if args.output_dir else grid_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_summary(summary_path)
    variants = ["source"] + [f"strength_{strength:g}" for strength in args.strengths]
    nrow = 1 + len(variants)

    grid = Image.open(grid_path)
    cell_size = infer_cell_size(grid.width, nrow=nrow, padding=args.padding)
    expected_height = args.padding + len(summary_rows) * (cell_size + args.padding)
    if grid.height != expected_height:
        raise ValueError(
            f"Unexpected grid height: got {grid.height}, expected {expected_height} "
            f"for {len(summary_rows)} rows, cell_size={cell_size}, padding={args.padding}"
        )

    records = []
    clip_gt_images = []
    clip_pred_images = []
    clip_record_indices = []

    for row_idx, row_meta in tqdm(list(enumerate(summary_rows)), desc="SSIM/MSE"):
        gt_img = crop_cell(grid, row_idx, 0, cell_size, args.padding)
        for variant_idx, variant in enumerate(variants, start=1):
            pred_img = crop_cell(grid, row_idx, variant_idx, cell_size, args.padding)
            record = {
                "row": row_meta["row"],
                "query_original_idx": row_meta["query_original_idx"],
                "source_original_idx": row_meta["source_original_idx"],
                "self_match": row_meta["self_match"],
                "variant": variant,
                "ssim": compute_ssim(gt_img, pred_img),
                "mse": compute_mse(gt_img, pred_img),
                "clip_similarity": float("nan"),
            }
            records.append(record)
            if args.compute_clip:
                clip_record_indices.append(len(records) - 1)
                clip_gt_images.append(gt_img)
                clip_pred_images.append(pred_img)

    if args.compute_clip:
        device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
        clip_metric = ClipImageSimilarity(args.clip_model_path, device)
        clip_scores = clip_metric.score_pairs(clip_gt_images, clip_pred_images, args.clip_batch_size)
        for record_idx, score in zip(clip_record_indices, clip_scores):
            records[record_idx]["clip_similarity"] = score

    stem = grid_path.stem
    metrics_csv = output_dir / f"{stem}_metrics_by_sample.csv"
    metrics_json = output_dir / f"{stem}_metrics_summary.json"

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(summarize(records), f, indent=2)

    print(f"[done] wrote {metrics_csv}")
    print(f"[done] wrote {metrics_json}")


if __name__ == "__main__":
    main()
