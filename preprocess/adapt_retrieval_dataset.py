#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_img_paths(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(object).copy()
    it = np.nditer(out, flags=["multi_index", "refs_ok"], op_flags=["readwrite"])
    for x in it:
        s = str(x.item()).replace("\\", "/")
        x[...] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-data-dir", type=Path, default=Path("/home/yiqiuliu/DL_Project/image-eeg-data"))
    ap.add_argument("--dst-root", type=Path, default=Path("/home/yiqiuliu/VisualEEGDecoding/data/things-eeg"))
    ap.add_argument("--subject-id", type=int, default=1)
    args = ap.parse_args()

    tr = torch.load(args.src_data_dir / "train.pt", map_location="cpu", weights_only=False)
    te = torch.load(args.src_data_dir / "test.pt", map_location="cpu", weights_only=False)

    dst = args.dst_root
    img_set = dst / "Image_set"
    preproc = dst / "Preprocessed_data" / f"sub-{args.subject_id:02d}"
    _ensure_dir(img_set)
    _ensure_dir(preproc)

    # Symlink image folders so preprocess/train scripts can resolve image keys.
    train_src = args.src_data_dir / "training_images"
    test_src = args.src_data_dir / "test_images"
    train_dst = img_set / "train_images"
    test_dst = img_set / "test_images"
    if not train_dst.exists():
        train_dst.symlink_to(train_src)
    if not test_dst.exists():
        test_dst.symlink_to(test_src)

    # Keep their expected keys and layout.
    tr_out = dict(tr)
    te_out = dict(te)
    tr_out["img"] = _normalize_img_paths(np.array(tr_out["img"]))
    te_out["img"] = _normalize_img_paths(np.array(te_out["img"]))

    torch.save(tr_out, preproc / "train.pt", pickle_protocol=5)
    torch.save(te_out, preproc / "test.pt", pickle_protocol=5)

    print(f"[ok] retrieval dataset adapted -> {preproc}")
    print(f"[ok] image symlinks -> {train_dst}, {test_dst}")
    print(f"[shape] train eeg={np.array(tr_out['eeg']).shape}, img={np.array(tr_out['img']).shape}")
    print(f"[shape] test  eeg={np.array(te_out['eeg']).shape}, img={np.array(te_out['img']).shape}")


if __name__ == "__main__":
    main()
