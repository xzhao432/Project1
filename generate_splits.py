"""
Generate splits file for DreamDiffusion.
Following VisualEEGDecoding approach: random split with seed.
"""

import torch
import numpy as np

def generate_splits(n_samples, train_ratio=0.9, seed=2025):
    """
    Generate random train/val split.

    Args:
        n_samples: Total number of samples
        train_ratio: Ratio of training samples (default 0.9)
        seed: Random seed for reproducibility

    Returns:
        dict with 'splits' key containing train/test indices
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    train_idx = sorted(indices[:n_train].tolist())
    test_idx = sorted(indices[n_train:].tolist())

    return {
        'splits': [{
            'train': train_idx,
            'test': test_idx
        }]
    }

if __name__ == '__main__':
    # Load train data to get number of samples
    train_data = torch.load('/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion.pt', weights_only=False)
    n_samples = len(train_data['dataset'])

    print("="*80)
    print("Generating splits file")
    print("="*80)
    print(f"Total samples: {n_samples}")
    print(f"Train ratio: 0.9 (90%)")
    print(f"Val ratio: 0.1 (10%)")
    print(f"Random seed: 2025")

    splits = generate_splits(n_samples, train_ratio=0.9, seed=2025)

    n_train = len(splits['splits'][0]['train'])
    n_val = len(splits['splits'][0]['test'])

    print(f"\nGenerated splits:")
    print(f"  Train: {n_train} samples ({n_train/n_samples*100:.1f}%)")
    print(f"  Val: {n_val} samples ({n_val/n_samples*100:.1f}%)")

    # Save splits
    output_path = '/home/yiqiuliu/DL_Project/image-eeg-data/block_splits_random.pth'
    torch.save(splits, output_path)
    print(f"\nSaved to: {output_path}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print("Next steps:")
    print("1. Update precompute script to use train_dreamdiffusion.pt and block_splits_random.pth")
    print("2. Update training script to use train_dreamdiffusion.pt and block_splits_random.pth")
