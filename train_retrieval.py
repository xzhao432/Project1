#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_image_ids(img_field) -> list[str]:
    arr = np.array(img_field)
    if arr.ndim == 2:
        base = arr[:, 0].tolist()
    else:
        base = arr.tolist()
    out = []
    for x in base:
        s = str(x).replace("\\", "/")
        out.append(Path(s).stem)
    return out


def _avg_eeg(eeg_field) -> np.ndarray:
    x = torch.as_tensor(eeg_field, dtype=torch.float32)
    # [N, R, C, T] -> [N, C, T]
    if x.ndim == 4:
        x = x.mean(dim=1)
    if x.ndim != 3:
        raise ValueError(f"Expected EEG [N,C,T] or [N,R,C,T], got {tuple(x.shape)}")
    return x.cpu().numpy().astype(np.float32)


class PairDataset(Dataset):
    def __init__(self, eeg: np.ndarray, ids: list[str], clip_feats: torch.Tensor, id_to_idx: dict[str, int]):
        self.eeg = torch.from_numpy(eeg).float()
        self.ids = ids
        self.clip_feats = clip_feats
        self.id_to_idx = id_to_idx

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        x = self.eeg[idx]
        key = self.ids[idx]
        clip_idx = self.id_to_idx[key]
        img = self.clip_feats[clip_idx]  # [L, D]
        return {"eeg": x, "img_list": img, "x_key": key}


@torch.no_grad()
def eval_topk(model, loader, device):
    model.eval()
    total = 0
    t1 = t3 = t5 = 0
    ranks = []
    for batch in loader:
        x = batch["eeg"].to(device)
        img_list = batch["img_list"].to(device)
        ze = model(x)
        ze = ze / ze.norm(dim=-1, keepdim=True)
        zi = model.get_image_feature(img_list)
        zi = zi / zi.norm(dim=-1, keepdim=True)
        sim = ze @ zi.T
        topk = sim.topk(k=min(5, sim.shape[1]), dim=1).indices.cpu()
        y = torch.arange(sim.shape[0]).unsqueeze(1)
        t1 += (topk[:, :1] == y).any(dim=1).sum().item()
        t3 += (topk[:, :3] == y).any(dim=1).sum().item()
        t5 += (topk[:, :5] == y).any(dim=1).sum().item()

        # Calculate ranks for each sample
        for i in range(sim.shape[0]):
            rank = (sim[i] >= sim[i, i]).sum().item()
            ranks.append(rank)

        total += sim.shape[0]

    mean_rank = np.mean(ranks) if ranks else 0
    median_rank = np.median(ranks) if ranks else 0
    return t1 / total, t3 / total, t5 / total, mean_rank, median_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/home/yiqiuliu/DL_Project/image-eeg-data"))
    ap.add_argument("--clip-features", type=Path, default=Path("/home/yiqiuliu/VisualEEGDecoding/data/things-eeg/Image_feature/MultiBlur_RN50_train.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("./runs/retrieval"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    ap.add_argument("--use-test-as-val", action="store_true", help="Use test set as validation (not recommended)")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW")
    ap.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping patience")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    encoder_path = Path(__file__).resolve().parent / "models" / "Encoder.py"
    spec = importlib.util.spec_from_file_location("visualeeg_encoder", encoder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load encoder from {encoder_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    train_pt = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=False)
    test_pt = torch.load(args.data_dir / "test.pt", map_location="cpu", weights_only=False)
    eeg_train = _avg_eeg(train_pt["eeg"])
    eeg_test = _avg_eeg(test_pt["eeg"])
    ids_train = _to_image_ids(train_pt["img"])
    ids_test = _to_image_ids(test_pt["img"])

    # Select visual cortex channels (posterior regions) as in the paper
    ALL_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                    'O1', 'Oz', 'O2']
    VISUAL_CHANNELS = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                       'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
    selected_idx = [ALL_CHANNELS.index(ch) for ch in VISUAL_CHANNELS]
    eeg_train = eeg_train[:, selected_idx, :]
    eeg_test = eeg_test[:, selected_idx, :]

    # Load MultiBlur_RN50 features (dict format with blur levels as keys)
    cache = torch.load(args.clip_features, map_location="cpu", weights_only=False)

    # Always load test features (needed for test set evaluation)
    test_feat_path = args.clip_features.parent / "MultiBlur_RN50_test.pt"
    if test_feat_path.exists():
        cache_test = torch.load(test_feat_path, map_location="cpu", weights_only=False)
        print(f"[info] Loaded test features from {test_feat_path}")
    else:
        raise FileNotFoundError(f"Test features not found at {test_feat_path}")

    # Check if it's the new MultiBlur format (dict with blur level keys) or old format
    if isinstance(cache, dict) and "1" in cache:
        # MultiBlur_RN50 format: {'1': {img_path: feat}, '3': {...}, ...}
        blur_levels = ["1", "3", "9", "15", "21", "27", "33", "39", "45", "51", "57", "63"]

        # Collect all keys from train and test
        train_keys = sorted(cache["1"].keys())
        test_keys = sorted(cache_test["1"].keys())
        all_keys = train_keys + test_keys

        # Build id_to_idx mapping
        id_to_idx = {}
        for i, key in enumerate(all_keys):
            stem = Path(key).stem
            id_to_idx[stem] = i

        # Stack features: [N_train + N_test, 12, 1024]
        n_imgs = len(all_keys)
        feat_dim = cache["1"][train_keys[0]].shape[0]
        clip_feats = torch.zeros(n_imgs, len(blur_levels), feat_dim, dtype=torch.float32)

        # Load train features
        for level_idx, level in enumerate(blur_levels):
            for img_idx, key in enumerate(train_keys):
                clip_feats[img_idx, level_idx] = cache[level][key]

        # Load test features if available
        offset = len(train_keys)
        for level_idx, level in enumerate(blur_levels):
            for img_idx, key in enumerate(test_keys):
                clip_feats[offset + img_idx, level_idx] = cache_test[level][key]

        clip_feats = F.normalize(clip_feats, dim=-1)
        print(f"[info] Loaded MultiBlur_RN50 features: {clip_feats.shape} ({len(train_keys)} train + {len(test_keys)} test), {len(id_to_idx)} unique image stems")
    else:
        # Old format: {'features': [N, L, D], 'id_to_idx': {...}}
        clip_feats = torch.as_tensor(cache["features"], dtype=torch.float32)
        id_to_idx = cache["id_to_idx"]
        if clip_feats.ndim == 2:
            clip_feats = clip_feats[:, None, :]
        clip_feats = F.normalize(clip_feats, dim=-1)
        # paper uses 12 blur levels; adapt to model input by clipping/padding to 12.
        target_levels = 12
        l = clip_feats.shape[1]
        if l >= target_levels:
            clip_feats = clip_feats[:, :target_levels, :]
        else:
            pad = clip_feats[:, -1:, :].repeat(1, target_levels - l, 1)
            clip_feats = torch.cat([clip_feats, pad], dim=1)

    # Split train into train/val
    n = len(ids_train)
    idx = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(n * args.val_ratio)))
    tr_idx = idx[n_val:]
    va_idx = idx[:n_val]

    eeg_tr = eeg_train[tr_idx]
    ids_tr = [ids_train[i] for i in tr_idx]
    eeg_va = eeg_train[va_idx]
    ids_va = [ids_train[i] for i in va_idx]

    # Override with test set if requested
    if args.use_test_as_val:
        print("[warning] Using test set as validation - not recommended!")
        eeg_va, ids_va = eeg_test, ids_test

    if args.max_train_samples > 0:
        eeg_tr = eeg_tr[: args.max_train_samples]
        ids_tr = ids_tr[: args.max_train_samples]

    print(f"[info] Train: {len(ids_tr)}, Val: {len(ids_va)}, Test: {len(ids_test)}")
    print(f"[info] Temperature: {args.temperature}, Weight decay: {args.weight_decay}")

    ch = eeg_tr.shape[1]
    t = eeg_tr.shape[2]
    dim = clip_feats.shape[-1]

    model = mod.Brain_Visual_Encoder_EEG(channels=ch, proj_dim=dim, temporal_len=t).to(device)
    print(f"[info] Using Brain_Visual_Encoder_EEG with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Use AdamW with weight decay
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(PairDataset(eeg_tr, ids_tr, clip_feats, id_to_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(PairDataset(eeg_va, ids_va, clip_feats, id_to_idx), batch_size=min(len(ids_va), 512), shuffle=False)
    test_loader = DataLoader(PairDataset(eeg_test, ids_test, clip_feats, id_to_idx), batch_size=min(len(ids_test), 512), shuffle=False)

    metrics_path = args.out_dir / "epoch_metrics.jsonl"
    best_val_top1 = 0.0
    patience_counter = 0

    for e in range(args.epochs):
        model.train()
        losses = []
        pos_sims = []
        neg_sims = []
        diag_sims = []
        offdiag_sims = []

        for batch in tqdm(train_loader, desc=f"epoch {e+1}/{args.epochs}"):
            x = batch["eeg"].to(device)
            img_list = batch["img_list"].to(device)
            ze = model(x)
            zi = model.get_image_feature(img_list)
            ze = ze / ze.norm(dim=-1, keepdim=True)
            zi = zi / zi.norm(dim=-1, keepdim=True)
            logits = ze @ zi.T

            # Apply temperature scaling
            logits = logits / args.temperature

            y = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))

            # Track similarity statistics (before temperature scaling)
            with torch.no_grad():
                logits_raw = (ze @ zi.T)  # Raw similarities without temperature
                # Positive pairs (diagonal)
                pos_sim = torch.diagonal(logits_raw).mean().item()
                pos_sims.append(pos_sim)

                # Negative pairs (off-diagonal)
                mask = ~torch.eye(logits_raw.shape[0], dtype=torch.bool, device=device)
                neg_sim = logits_raw[mask].mean().item()
                neg_sims.append(neg_sim)

                # Diagonal vs off-diagonal
                diag_sims.append(pos_sim)
                offdiag_sims.append(neg_sim)

        # Evaluate on fixed training subset (for fair comparison with val/test)
        train_eval_loader = DataLoader(
            PairDataset(eeg_tr[:min(1000, len(eeg_tr))], ids_tr[:min(1000, len(ids_tr))], clip_feats, id_to_idx),
            batch_size=min(512, len(ids_tr)), shuffle=False
        )
        train_t1, train_t3, train_t5, train_mean_rank, train_median_rank = eval_topk(model, train_eval_loader, device)
        val_t1, val_t3, val_t5, val_mean_rank, val_median_rank = eval_topk(model, val_loader, device)
        test_t1, test_t3, test_t5, test_mean_rank, test_median_rank = eval_topk(model, test_loader, device)

        # Save best model based on val_top1
        if val_t1 > best_val_top1:
            best_val_top1 = val_t1
            torch.save(model.state_dict(), args.out_dir / "best.pth")
            print(f"[best] Saved best model at epoch {e+1} with val_top1={val_t1:.3f}")
            patience_counter = 0
        else:
            patience_counter += 1

        row = {
            "epoch": e + 1,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_loss_std": float(np.std(losses)) if losses else None,
            "pos_sim_mean": float(np.mean(pos_sims)) if pos_sims else None,
            "neg_sim_mean": float(np.mean(neg_sims)) if neg_sims else None,
            "margin": float(np.mean(pos_sims) - np.mean(neg_sims)) if pos_sims and neg_sims else None,
            "diag_mean": float(np.mean(diag_sims)) if diag_sims else None,
            "offdiag_mean": float(np.mean(offdiag_sims)) if offdiag_sims else None,
            "train_eval_top1": float(train_t1),
            "train_eval_top3": float(train_t3),
            "train_eval_top5": float(train_t5),
            "train_mean_rank": float(train_mean_rank),
            "train_median_rank": float(train_median_rank),
            "val_top1": float(val_t1),
            "val_top3": float(val_t3),
            "val_top5": float(val_t5),
            "val_mean_rank": float(val_mean_rank),
            "val_median_rank": float(val_median_rank),
            "test_top1": float(test_t1),
            "test_top3": float(test_t3),
            "test_top5": float(test_t5),
            "test_mean_rank": float(test_mean_rank),
            "test_median_rank": float(test_median_rank),
        }
        print(f"Epoch {e+1}: train_top1={train_t1:.3f}, val_top1={val_t1:.3f}, test_top1={test_t1:.3f}, margin={row['margin']:.3f}")
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"[early stop] No improvement for {args.early_stop_patience} epochs. Stopping at epoch {e+1}.")
            break

    torch.save(model.state_dict(), args.out_dir / "last.pth")
    print(f"[done] Saved to {args.out_dir}")
    print(f"[done] Best val_top1: {best_val_top1:.3f}")


if __name__ == "__main__":
    main()
