#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


BLUR_TAGS = ["1", "3", "9", "15", "21", "27", "33", "39", "45", "51", "57", "63"]
BLUR_KERNELS = [1, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63]


def blur_image(img: Image.Image, k: int) -> Image.Image:
    if k <= 1:
        return img
    arr = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    out = cv2.GaussianBlur(bgr, (k, k), 0)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def collect_rel_paths(root: Path) -> list[str]:
    out: list[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            out.append(str(p.relative_to(root.parent)).replace("\\", "/"))
    out.sort()
    return out


@torch.no_grad()
def encode_paths(paths: list[str], image_root: Path, model, preproc, k: int, batch_size: int, device: torch.device, checkpoint_file: Path | None = None):
    feats = {}
    start_idx = 0

    # Try to resume from checkpoint
    if checkpoint_file and checkpoint_file.exists():
        try:
            ckpt = torch.load(checkpoint_file, map_location="cpu")
            feats = ckpt["features"]
            start_idx = ckpt["next_idx"]
            print(f"[info] Resuming blur={k} from index {start_idx}/{len(paths)}")
        except Exception as e:
            print(f"[warn] Failed to load checkpoint: {e}, starting from scratch")
            feats = {}
            start_idx = 0

    checkpoint_every = 100
    for i in tqdm(range(start_idx, len(paths), batch_size), desc=f"blur={k}"):
        batch_paths = paths[i : i + batch_size]
        imgs = []
        for rel in batch_paths:
            p = image_root.parent / rel
            img = Image.open(p).convert("RGB").resize((224, 224))
            img = blur_image(img, k)
            imgs.append(preproc(img))
        x = torch.stack(imgs, dim=0).to(device)
        z = model.encode_image(x)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.float().cpu()
        for rel, f in zip(batch_paths, z):
            feats[rel] = f

        # Save checkpoint periodically
        if checkpoint_file and (i - start_idx) % (checkpoint_every * batch_size) == 0 and i > start_idx:
            torch.save({"features": feats, "next_idx": i + batch_size}, checkpoint_file)

    # Clean up checkpoint on success
    if checkpoint_file and checkpoint_file.exists():
        checkpoint_file.unlink()

    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/home/yiqiuliu/VisualEEGDecoding/data/things-eeg"))
    ap.add_argument("--clip-weights", type=str, default="")
    ap.add_argument(
        "--backend",
        choices=("open_clip", "clip"),
        default="open_clip",
        help="open_clip: RN50 open_clip weights; clip: OpenAI CLIP TorchScript .pt",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--split", choices=("all", "train", "test"), default="all")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.backend == "open_clip":
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "RN50", pretrained=args.clip_weights if args.clip_weights else "openai"
        )
        model = model.to(device).eval()
        preproc = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        import clip

        if not args.clip_weights:
            raise ValueError("--backend clip requires --clip-weights /path/to/RN50.pt")
        model, _ = clip.load(args.clip_weights, device=device)
        model = model.to(device).eval()
        preproc = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    image_set = args.data_root / "Image_set"
    train_root = image_set / "train_images"
    test_root = image_set / "test_images"
    train_paths = collect_rel_paths(train_root) if args.split in {"all", "train"} else []
    test_paths = collect_rel_paths(test_root) if args.split in {"all", "test"} else []
    if args.limit_train > 0:
        train_paths = train_paths[: args.limit_train]
    if args.limit_test > 0:
        test_paths = test_paths[: args.limit_test]

    out_dir = args.data_root / "Image_feature"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_dir / "MultiBlur_RN50_train.pt"
    out_test = out_dir / "MultiBlur_RN50_test.pt"

    # Checkpoint files
    ckpt_train_prefix = out_dir / ".MultiBlur_RN50_train_ckpt"
    ckpt_test_prefix = out_dir / ".MultiBlur_RN50_test_ckpt"

    train_saved = {}
    test_saved = {}
    for tag, k in zip(BLUR_TAGS, BLUR_KERNELS):
        ckpt_train = ckpt_train_prefix.parent / f"{ckpt_train_prefix.name}_{tag}.pt"
        ckpt_test = ckpt_test_prefix.parent / f"{ckpt_test_prefix.name}_{tag}.pt"
        if args.split in {"all", "train"}:
            train_saved[tag] = encode_paths(train_paths, train_root, model, preproc, k, args.batch_size, device, ckpt_train)
        if args.split in {"all", "test"}:
            test_saved[tag] = encode_paths(test_paths, test_root, model, preproc, k, args.batch_size, device, ckpt_test)
    if args.split in {"all", "train"}:
        torch.save(train_saved, out_train)
        print(f"[ok] saved {out_train}")
    if args.split in {"all", "test"}:
        torch.save(test_saved, out_test)
        print(f"[ok] saved {out_test}")


if __name__ == "__main__":
    main()
