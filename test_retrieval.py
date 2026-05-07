#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ALL_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                'O1', 'Oz', 'O2']
VISUAL_CHANNELS = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                   'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
BLUR_LEVELS = ['1', '3', '9', '15', '21', '27', '33', '39', '45', '51', '57', '63']


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_image_ids(img_field) -> list[str]:
    arr = np.array(img_field)
    base = arr[:, 0].tolist() if arr.ndim == 2 else arr.tolist()
    return [Path(str(x).replace('\\', '/')).stem for x in base]


def _avg_eeg(eeg_field) -> np.ndarray:
    x = torch.as_tensor(eeg_field, dtype=torch.float32)
    if x.ndim == 4:
        x = x.mean(dim=1)
    if x.ndim != 3:
        raise ValueError(f'Expected EEG [N,C,T] or [N,R,C,T], got {tuple(x.shape)}')
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
        key = self.ids[idx]
        return {'eeg': self.eeg[idx], 'img_list': self.clip_feats[self.id_to_idx[key]], 'x_key': key}


def load_eeg_split(data_dir: Path, split: str) -> tuple[np.ndarray, list[str]]:
    pt = torch.load(data_dir / f'{split}.pt', map_location='cpu', weights_only=False)
    eeg = _avg_eeg(pt['eeg'])
    ids = _to_image_ids(pt['img'])
    selected_idx = [ALL_CHANNELS.index(ch) for ch in VISUAL_CHANNELS]
    return eeg[:, selected_idx, :], ids


def _resolve_feature_path(feature_path: Path, split: str) -> Path:
    expected_name = f'MultiBlur_RN50_{split}.pt'
    if feature_path.name in {'MultiBlur_RN50_train.pt', 'MultiBlur_RN50_test.pt'}:
        candidate = feature_path.parent / expected_name
        if candidate.exists():
            return candidate
    return feature_path


def load_multiblur_features(feature_path: Path, split: str) -> tuple[torch.Tensor, dict[str, int]]:
    cache_path = _resolve_feature_path(feature_path, split)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    keys = sorted(cache['1'].keys())
    if not keys:
        raise ValueError(f'No image features found in {cache_path}')

    id_to_idx = {Path(key).stem: i for i, key in enumerate(keys)}
    feat_dim = cache['1'][keys[0]].shape[0]
    clip_feats = torch.zeros(len(keys), len(BLUR_LEVELS), feat_dim, dtype=torch.float32)
    for level_idx, level in enumerate(BLUR_LEVELS):
        for img_idx, key in enumerate(keys):
            clip_feats[img_idx, level_idx] = cache[level][key]
    return F.normalize(clip_feats, dim=-1), id_to_idx


def load_encoder(channels: int, proj_dim: int, temporal_len: int, checkpoint: Path, device: torch.device):
    encoder_path = Path(__file__).resolve().parent / 'models' / 'Encoder.py'
    spec = importlib.util.spec_from_file_location('visualeeg_encoder', encoder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load encoder from {encoder_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Brain_Visual_Encoder_EEG(channels=channels, proj_dim=proj_dim, temporal_len=temporal_len).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def encode_split(model, loader: DataLoader, device: torch.device):
    eeg_chunks = []
    img_chunks = []
    ids = []
    for batch in tqdm(loader, desc='encode'):
        x = batch['eeg'].to(device)
        img_list = batch['img_list'].to(device)
        ze = model(x)
        zi = model.get_image_feature(img_list)
        eeg_chunks.append(ze.cpu())
        img_chunks.append(zi.cpu())
        ids.extend(batch['x_key'])
    return ids, torch.cat(eeg_chunks, dim=0), torch.cat(img_chunks, dim=0)


def write_rankings(out_path: Path, ids: list[str], sim: torch.Tensor, k: int) -> None:
    topk = sim.topk(k=min(k, sim.shape[1]), dim=1).indices
    with out_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['query_id', 'target_id', 'rank'] + [f'top{i + 1}' for i in range(topk.shape[1])]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(topk.tolist()):
            rank = int((sim[i] >= sim[i, i]).sum().item())
            rec = {'query_id': ids[i], 'target_id': ids[i], 'rank': rank}
            rec.update({f'top{j + 1}': ids[idx] for j, idx in enumerate(row)})
            writer.writerow(rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path('/home/yiqiuliu/DL_Project/image-eeg-data'))
    ap.add_argument('--clip-features', type=Path, default=Path('/home/yiqiuliu/VisualEEGDecoding/data/things-eeg/Image_feature/MultiBlur_RN50_test.pt'))
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('./outputs/retrieval_inference'))
    ap.add_argument('--split', choices=['train', 'test'], default='test')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2025)
    args = ap.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eeg, ids = load_eeg_split(args.data_dir, args.split)
    clip_feats, id_to_idx = load_multiblur_features(args.clip_features, args.split)
    model = load_encoder(eeg.shape[1], clip_feats.shape[-1], eeg.shape[2], args.checkpoint, device)

    loader = DataLoader(PairDataset(eeg, ids, clip_feats, id_to_idx), batch_size=args.batch_size, shuffle=False)
    ids, eeg_features, img_features = encode_split(model, loader, device)
    eeg_norm = F.normalize(eeg_features, dim=-1)
    img_norm = F.normalize(img_features, dim=-1)
    sim = eeg_norm @ img_norm.T

    labels = torch.arange(sim.shape[0])
    topk = sim.topk(k=min(5, sim.shape[1]), dim=1).indices
    top1 = float((topk[:, :1] == labels[:, None]).any(dim=1).float().mean().item())
    top3 = float((topk[:, :3] == labels[:, None]).any(dim=1).float().mean().item())
    top5 = float((topk[:, :5] == labels[:, None]).any(dim=1).float().mean().item())
    ranks = torch.tensor([(sim[i] >= sim[i, i]).sum().item() for i in range(sim.shape[0])], dtype=torch.float32)
    metrics = {
        'split': args.split,
        'num_samples': len(ids),
        'top1': top1,
        'top3': top3,
        'top5': top5,
        'mean_rank': float(ranks.mean().item()),
        'median_rank': float(ranks.median().item()),
    }

    feature_path = args.out_dir / f'features_{args.split}.pt'
    torch.save({
        'ids': ids,
        'eeg_output': {key: eeg_features[i] for i, key in enumerate(ids)},
        'img_output': {key: img_features[i] for i, key in enumerate(ids)},
    }, feature_path)
    write_rankings(args.out_dir / f'rankings_{args.split}.csv', ids, sim, args.top_k)
    with (args.out_dir / f'metrics_{args.split}.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f'[done] wrote {feature_path}')
    print(f'[done] top1={top1:.4f}, top3={top3:.4f}, top5={top5:.4f}, mean_rank={metrics["mean_rank"]:.2f}')


if __name__ == '__main__':
    main()
