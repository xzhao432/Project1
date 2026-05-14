"""
Precompute VAE latents and image supervision features for DreamDiffusion training.
This significantly speeds up training by computing fixed features once.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
from pathlib import Path
import h5py

# Add code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from dataset import EEGDataset, Splitter
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from transformers import CLIPModel, AutoProcessor
import torchvision.transforms as transforms


def get_vae_model(pretrain_path, device):
    """Load VAE model from pretrained checkpoint."""
    config_path = os.path.join(pretrain_path, 'models/config15.yaml')
    ckpt_path = os.path.join(pretrain_path, 'models/eeg_pretrain_scp/v1-5-pruned.ckpt')

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)['state_dict']
    model.load_state_dict(pl_sd, strict=False)

    vae = model.first_stage_model.to(device)
    vae.eval()

    # Get scale_factor from the model
    scale_factor = model.scale_factor
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.item()

    print(f"VAE scale_factor: {scale_factor}")

    return vae, scale_factor


def get_clip_model(device, pretrain_path):
    """Load CLIP model for image features."""
    # Use the old pretrain directory where CLIP model is located
    clip_path = '/home/yiqiuliu/DreamDiffusion_old/pretrains/models/eeg_pretrain_scp/clip_vit_large_patch14'

    # Try to load from local path first
    if os.path.exists(clip_path):
        print(f"Loading CLIP from local path: {clip_path}")
        model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device)
        processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    else:
        print("Loading CLIP from HuggingFace...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model.eval()
    return model, processor


def compute_color_histogram(image, bins=32):
    """Compute color histogram features."""
    # image: (C, H, W) tensor in range [-1, 1]
    image = (image + 1) / 2  # to [0, 1]
    image = (image * 255).byte()

    histograms = []
    for c in range(3):
        hist = torch.histc(image[c].float(), bins=bins, min=0, max=255)
        hist = hist / hist.sum()
        histograms.append(hist)

    return torch.cat(histograms)


def compute_brightness_contrast(image):
    """Compute brightness and contrast features."""
    # image: (C, H, W) tensor in range [-1, 1]
    image = (image + 1) / 2  # to [0, 1]

    brightness = image.mean()
    contrast = image.std()

    return torch.tensor([brightness, contrast])


def compute_texture_features(image):
    """Compute simple texture features (edge statistics)."""
    # image: (C, H, W) tensor in range [-1, 1]
    gray = image.mean(dim=0)  # Convert to grayscale

    # Simple edge detection (Sobel-like)
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]

    edge_mean = (dx.abs().mean() + dy.abs().mean()) / 2
    edge_std = (dx.std() + dy.std()) / 2

    return torch.tensor([edge_mean, edge_std])


@torch.no_grad()
def precompute_dataset(
    eeg_signals_path,
    splits_path,
    imagenet_path,
    pretrain_path,
    output_path,
    subject=4,
    batch_size=8,
    device='cuda'
):
    """
    Precompute all features for the dataset.

    Args:
        eeg_signals_path: Path to EEG signals .pt file
        splits_path: Path to dataset splits .pth file
        imagenet_path: Path to ImageNet images
        pretrain_path: Path to pretrained models
        output_path: Where to save precomputed features
        subject: Subject number
        batch_size: Batch size for processing (larger = faster)
        device: 'cuda' or 'cpu'
    """

    print(f"Precomputing features for subject {subject}")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load models
    print("\nLoading models...")
    vae, scale_factor = get_vae_model(pretrain_path, device)
    clip_model, clip_processor = get_clip_model(device, pretrain_path)

    # Image transform
    def image_transform(img):
        # img is numpy array (H, W, 3) in range [0, 1]
        from einops import rearrange
        from PIL import Image
        import torchvision.transforms as transforms

        # Convert to PIL Image for resizing
        img_pil = Image.fromarray((img * 255).astype('uint8'))

        # Resize to 512x512
        img_pil = transforms.Resize((512, 512))(img_pil)

        # Convert back to tensor
        img = torch.tensor(np.array(img_pil) / 255.0)

        # Rearrange to (C, H, W)
        if img.shape[-1] == 3:
            img = rearrange(img, 'h w c -> c h w')

        # Normalize to [-1, 1]
        img = img * 2.0 - 1.0
        return img

    # Load dataset
    print("\nLoading dataset...")
    dataset_full = EEGDataset(eeg_signals_path, imagenet_path, image_transform, subject)

    # Load splits
    loaded_splits = torch.load(splits_path)
    train_idx = loaded_splits["splits"][0]["train"]
    test_idx = loaded_splits["splits"][0]["test"]

    # Filter indices - keep samples with reasonable time dimensions
    # DreamDiffusion expects ~512 time points, so we filter for 450-600 range
    train_idx = [i for i in train_idx if i < len(dataset_full.data) and 450 <= dataset_full.data[i]["eeg"].size(1) <= 600]
    test_idx = [i for i in test_idx if i < len(dataset_full.data) and 450 <= dataset_full.data[i]["eeg"].size(1) <= 600]

    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")

    # Process each split
    for split_name, indices in [('train', train_idx), ('test', test_idx)]:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split ({len(indices)} samples)")
        print(f"{'='*60}")

        output_file = os.path.join(output_path, f'{split_name}_precomputed.h5')

        # Create HDF5 file
        with h5py.File(output_file, 'w') as f:
            # Pre-allocate datasets
            n_samples = len(indices)
            f.create_dataset('vae_latents', shape=(n_samples, 4, 64, 64), dtype='float32')
            f.create_dataset('clip_features', shape=(n_samples, 768), dtype='float32')
            f.create_dataset('color_hist', shape=(n_samples, 96), dtype='float32')  # 32 bins * 3 channels
            f.create_dataset('brightness_contrast', shape=(n_samples, 2), dtype='float32')
            f.create_dataset('texture', shape=(n_samples, 2), dtype='float32')
            f.create_dataset('indices', data=np.array(indices), dtype='int32')

            # Process in batches
            for batch_start in tqdm(range(0, len(indices), batch_size), desc=f"Precomputing {split_name}"):
                batch_end = min(batch_start + batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]

                # Load batch data
                batch_images = []
                batch_images_raw = []

                for idx in batch_indices:
                    sample = dataset_full[idx]
                    batch_images.append(sample['image'])
                    batch_images_raw.append(sample['image_raw']['pixel_values'])

                batch_images = torch.stack(batch_images).to(device).float()
                batch_images_raw = torch.stack(batch_images_raw).to(device).float()

                # Compute VAE latents
                vae_latents = vae.encode(batch_images).sample()

                # Apply scale_factor (same as in ddpm.py get_first_stage_encoding)
                vae_latents = vae_latents * scale_factor

                # Debug: print statistics for first batch
                if batch_start == 0:
                    print(f"\n[Debug] First batch VAE latent statistics:")
                    print(f"  Shape: {vae_latents.shape}")
                    print(f"  Mean: {vae_latents.mean().item():.6f}")
                    print(f"  Std: {vae_latents.std().item():.6f}")
                    print(f"  Min: {vae_latents.min().item():.6f}")
                    print(f"  Max: {vae_latents.max().item():.6f}")

                # Compute CLIP features
                clip_features = clip_model.get_image_features(pixel_values=batch_images_raw)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

                # Compute image supervision features
                color_hists = []
                bc_features = []
                texture_features = []

                for img in batch_images:
                    color_hists.append(compute_color_histogram(img.cpu()))
                    bc_features.append(compute_brightness_contrast(img.cpu()))
                    texture_features.append(compute_texture_features(img.cpu()))

                color_hists = torch.stack(color_hists)
                bc_features = torch.stack(bc_features)
                texture_features = torch.stack(texture_features)

                # Save to HDF5
                f['vae_latents'][batch_start:batch_end] = vae_latents.cpu().numpy()
                f['clip_features'][batch_start:batch_end] = clip_features.cpu().numpy()
                f['color_hist'][batch_start:batch_end] = color_hists.numpy()
                f['brightness_contrast'][batch_start:batch_end] = bc_features.numpy()
                f['texture'][batch_start:batch_end] = texture_features.numpy()

        print(f"Saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Precompute features for DreamDiffusion')
    parser.add_argument('--eeg_signals_path', type=str, required=True,
                        help='Path to EEG signals .pt file')
    parser.add_argument('--splits_path', type=str, required=True,
                        help='Path to dataset splits .pth file')
    parser.add_argument('--imagenet_path', type=str, required=True,
                        help='Path to ImageNet images')
    parser.add_argument('--pretrain_path', type=str, required=True,
                        help='Path to pretrained models directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for precomputed features')
    parser.add_argument('--subject', type=int, default=4,
                        help='Subject number')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing (larger = faster)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    precompute_dataset(
        eeg_signals_path=args.eeg_signals_path,
        splits_path=args.splits_path,
        imagenet_path=args.imagenet_path,
        pretrain_path=args.pretrain_path,
        output_path=args.output_path,
        subject=args.subject,
        batch_size=args.batch_size,
        device=args.device
    )

    print("\n" + "="*60)
    print("Precomputation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
