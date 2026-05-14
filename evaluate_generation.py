#!/usr/bin/env python3
"""
Evaluation script for DreamDiffusion using original dataset with real ground truth images.
Computes SSIM, CLIP similarity, and other generation quality metrics.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim

# Add DreamDiffusion code to path
sys.path.insert(0, str(Path(__file__).parent / 'code'))

from dc_ldm.ldm_for_eeg import eLDM
from config import Config_Generative_Model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: Path, pretrain_root: Path, device: torch.device) -> eLDM:
    """Load trained DreamDiffusion model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', None)

    # Load pretrained MBM
    pretrain_mbm_path = pretrain_root / 'models' / 'eeg_pretrain_scp' / 'checkpoint.pth'
    pretrain_mbm_metafile = torch.load(pretrain_mbm_path, map_location='cpu', weights_only=False)

    # Get model configuration from checkpoint
    if config is not None:
        global_pool = getattr(config, 'global_pool', False)
        use_time_cond = getattr(config, 'use_time_cond', True)
        clip_tune = getattr(config, 'clip_tune', True)
        cls_tune = getattr(config, 'cls_tune', False)
        print(f"Using config from checkpoint: global_pool={global_pool}, use_time_cond={use_time_cond}, clip_tune={clip_tune}, cls_tune={cls_tune}")
    else:
        # Default values if config not found
        global_pool = False
        use_time_cond = True
        clip_tune = True
        cls_tune = False
        print("Warning: No config found in checkpoint, using default values")

    # Create model
    num_voxels = 512  # EEG time dimension
    model = eLDM(
        pretrain_mbm_metafile,
        num_voxels,
        device=device,
        pretrain_root=str(pretrain_root),
        logger=None,
        ddim_steps=250,
        global_pool=global_pool,
        use_time_cond=use_time_cond,
        clip_tune=clip_tune,
        cls_tune=cls_tune
    )

    # Load trained weights (strict=False to allow extra keys from checkpoint)
    missing_keys, unexpected_keys = model.model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint (will be ignored): {len(unexpected_keys)} keys")
    model.model.eval()

    print("Model loaded successfully")
    return model


def load_clip_model(device: torch.device, clip_path: Path):
    """Load CLIP model for computing similarity using Hugging Face transformers."""
    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading CLIP model from: {clip_path}")
    model = CLIPModel.from_pretrained(clip_path).to(device)
    processor = CLIPProcessor.from_pretrained(clip_path)

    return model, processor


def load_original_dataset(data_path: Path, imagenet_path: Path, subject: int = 0):
    """Load original EEG dataset with real images."""
    print(f"Loading original dataset from: {data_path}")

    # Load the data
    data = torch.load(data_path, weights_only=False)

    # Filter by subject
    if subject != 0:
        dataset = [item for item in data['dataset'] if item['subject'] == subject]
    else:
        dataset = data['dataset']

    print(f"Total samples for subject {subject}: {len(dataset)}")

    # Extract EEG, labels, and image paths
    test_eeg = []
    test_labels = []
    test_image_paths = []

    for item in dataset:
        test_eeg.append(item['eeg'])
        test_labels.append(item['label'])
        # Get image path from the images list
        image_idx = item['image']
        image_path = data['images'][image_idx]
        test_image_paths.append(image_path)

    test_eeg = torch.stack(test_eeg)
    test_labels = torch.tensor(test_labels)

    print(f"EEG shape: {test_eeg.shape}")
    print(f"Labels shape: {test_labels.shape}")
    print(f"Number of image paths: {len(test_image_paths)}")
    print(f"Example image path: {test_image_paths[0]}")

    return test_eeg, test_labels, test_image_paths, imagenet_path


@torch.no_grad()
def generate_and_evaluate(
    model: eLDM,
    test_eeg: torch.Tensor,
    test_labels: torch.Tensor,
    test_image_paths: list,
    imagenet_path: Path,
    clip_model,
    clip_processor,
    num_samples: int,
    ddim_steps: int,
    output_dir: Path,
    device: torch.device,
    limit: int = None
) -> dict:
    """
    Generate images and compute evaluation metrics.

    Returns:
        Dictionary containing all evaluation metrics
    """
    model.model.to(device)

    # Limit number of test samples if specified
    n_samples = min(limit, len(test_eeg)) if limit is not None else len(test_eeg)
    test_eeg = test_eeg[:n_samples]
    test_labels = test_labels[:n_samples]
    test_image_paths = test_image_paths[:n_samples]

    print(f"\nGenerating {num_samples} images for {n_samples} EEG samples...")
    print(f"DDIM steps: {ddim_steps}")

    # Load ground truth images
    print("Loading ground truth images...")
    gt_images = []
    for img_path in tqdm(test_image_paths, desc='Loading GT images'):
        full_path = imagenet_path / img_path
        img = Image.open(full_path).convert('RGB')
        # Resize to 512x512 to match generated images
        img = img.resize((512, 512), Image.LANCZOS)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        gt_images.append(img_array)

    # Prepare data in the format expected by model.generate()
    test_data = []
    for i in range(n_samples):
        # Keep HWC format as expected by model.generate()
        test_data.append({
            'eeg': test_eeg[i],
            'label': test_labels[i],
            'image': torch.from_numpy(gt_images[i]).float()  # Model expects HWC format
        })

    # Generate images
    grid, samples = model.generate(
        test_data,
        num_samples=num_samples,
        ddim_steps=ddim_steps,
        HW=(512, 512),
        limit=n_samples
    )

    # Save grid visualization
    grid_path = output_dir / 'evaluation_grid.png'
    Image.fromarray(grid.astype(np.uint8)).save(grid_path)
    print(f"Saved generation grid to: {grid_path}")

    # Compute metrics
    print("\nComputing evaluation metrics...")

    ssim_scores = []
    clip_similarities = []
    mse_scores = []

    images_dir = output_dir / 'evaluation_images'
    images_dir.mkdir(exist_ok=True)

    for sample_idx, imgs in enumerate(tqdm(samples, desc='Evaluating')):
        # imgs[0] is ground truth
        # imgs[1:] are generated samples
        gt_img = imgs[0]  # [C, H, W], already uint8 in [0, 255] from model.generate()
        gen_imgs = imgs[1:]  # List of [C, H, W]

        # Save ground truth (already uint8, no need to multiply by 255)
        gt_array = gt_img.astype(np.uint8)
        gt_pil = Image.fromarray(gt_array.transpose(1, 2, 0))  # CHW -> HWC
        gt_pil.save(images_dir / f'sample{sample_idx:04d}_gt.png')

        # Evaluate each generated image
        for gen_idx, gen_img in enumerate(gen_imgs):
            gen_array = gen_img.astype(np.uint8)
            gen_pil = Image.fromarray(gen_array.transpose(1, 2, 0))
            gen_pil.save(images_dir / f'sample{sample_idx:04d}_gen{gen_idx:02d}.png')

            # Compute SSIM (on grayscale or per-channel)
            # Convert to HWC format for SSIM
            gt_hwc = gt_img.transpose(1, 2, 0)  # [H, W, C], uint8 [0, 255]
            gen_hwc = gen_img.transpose(1, 2, 0)

            # Compute SSIM per channel and average (data_range=255 for uint8)
            ssim_val = ssim(
                gt_hwc,
                gen_hwc,
                data_range=255,
                channel_axis=2,
                win_size=11
            )
            ssim_scores.append(ssim_val)

            # Compute MSE (normalize to [0, 1] first for meaningful MSE)
            gt_normalized = gt_img.astype(np.float32) / 255.0
            gen_normalized = gen_img.astype(np.float32) / 255.0
            mse = np.mean((gt_normalized - gen_normalized) ** 2)
            mse_scores.append(mse)

            # Compute CLIP similarity
            # Prepare images for CLIP using processor
            inputs = clip_processor(images=[gt_pil, gen_pil], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get CLIP image features
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)

            # Split features for gt and gen
            gt_features = image_features[0:1]
            gen_features = image_features[1:2]

            # Normalize and compute cosine similarity
            gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)
            gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
            clip_sim = (gt_features * gen_features).sum().item()
            clip_similarities.append(clip_sim)

    # Aggregate metrics
    metrics = {
        'num_samples': n_samples,
        'num_generations_per_sample': num_samples,
        'total_generated_images': len(ssim_scores),
        'ssim': {
            'mean': float(np.mean(ssim_scores)),
            'std': float(np.std(ssim_scores)),
            'min': float(np.min(ssim_scores)),
            'max': float(np.max(ssim_scores)),
            'median': float(np.median(ssim_scores)),
        },
        'clip_similarity': {
            'mean': float(np.mean(clip_similarities)),
            'std': float(np.std(clip_similarities)),
            'min': float(np.min(clip_similarities)),
            'max': float(np.max(clip_similarities)),
            'median': float(np.median(clip_similarities)),
        },
        'mse': {
            'mean': float(np.mean(mse_scores)),
            'std': float(np.std(mse_scores)),
            'min': float(np.min(mse_scores)),
            'max': float(np.max(mse_scores)),
            'median': float(np.median(mse_scores)),
        }
    }

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate DreamDiffusion with real ground truth images')

    # Data paths
    ap.add_argument('--data-path', type=Path, required=True,
                    help='Path to original EEG dataset .pt file (e.g., train_for_dream.pt)')
    ap.add_argument('--imagenet-path', type=Path, required=True,
                    help='Path to ImageNet dataset directory')

    # Model paths
    ap.add_argument('--checkpoint', type=Path, required=True,
                    help='Path to trained model checkpoint')
    ap.add_argument('--pretrain-root', type=Path,
                    default=Path('/home/yiqiuliu/DreamDiffusion_old/pretrains'),
                    help='Path to pretrained models directory')

    # Generation parameters
    ap.add_argument('--num-samples', type=int, default=5,
                    help='Number of images to generate per EEG sample')
    ap.add_argument('--ddim-steps', type=int, default=250,
                    help='Number of DDIM sampling steps')
    ap.add_argument('--limit', type=int, default=None,
                    help='Limit number of test samples (for quick testing)')

    # Output
    ap.add_argument('--out-dir', type=Path, default=Path('./outputs/evaluation'))
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--subject', type=int, default=0)

    args = ap.parse_args()

    # Setup
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save configuration
    config_dict = vars(args).copy()
    config_dict['data_path'] = str(config_dict['data_path'])
    config_dict['imagenet_path'] = str(config_dict['imagenet_path'])
    config_dict['checkpoint'] = str(config_dict['checkpoint'])
    config_dict['pretrain_root'] = str(config_dict['pretrain_root'])
    config_dict['out_dir'] = str(config_dict['out_dir'])
    config_dict['device'] = str(device)
    config_dict['timestamp'] = datetime.now().isoformat()

    with (args.out_dir / 'eval_config.json').open('w') as f:
        json.dump(config_dict, f, indent=2)

    # Load original dataset
    test_eeg, test_labels, test_image_paths, imagenet_path = load_original_dataset(
        args.data_path, args.imagenet_path, args.subject
    )

    # Load model
    model = load_model(args.checkpoint, args.pretrain_root, device)

    # Load CLIP model
    print("\nLoading CLIP model...")
    clip_path = args.pretrain_root / 'models' / 'eeg_pretrain_scp' / 'clip_vit_large_patch14'
    clip_model, clip_processor = load_clip_model(device, clip_path)

    # Generate and evaluate
    metrics = generate_and_evaluate(
        model,
        test_eeg,
        test_labels,
        test_image_paths,
        imagenet_path,
        clip_model,
        clip_processor,
        args.num_samples,
        args.ddim_steps,
        args.out_dir,
        device,
        limit=args.limit
    )

    # Add config to metrics
    metrics['config'] = config_dict

    # Save metrics
    metrics_path = args.out_dir / 'evaluation_metrics.json'
    with metrics_path.open('w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"\nMetrics Summary:")
    print(f"  SSIM:            {metrics['ssim']['mean']:.4f} ± {metrics['ssim']['std']:.4f}")
    print(f"  CLIP Similarity: {metrics['clip_similarity']['mean']:.4f} ± {metrics['clip_similarity']['std']:.4f}")
    print(f"  MSE:             {metrics['mse']['mean']:.6f} ± {metrics['mse']['std']:.6f}")
    print(f"\nOutput directory: {args.out_dir}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
