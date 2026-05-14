"""
Convert our data format to original DreamDiffusion format.
Following VisualEEGDecoding approach: average 4 trials into 1 sample.

Our format:
{
  'eeg': (N, 4, 63, 250),  # N samples, 4 trials each
  'label': (N, 4),
  'img': (N, 4)  # image paths
}

Original DreamDiffusion format:
{
  'dataset': [
    {'eeg': (63, 250), 'label': int, 'image': int, 'subject': int},
    ...
  ],
  'labels': [class_ids],
  'images': [image_paths]
}

Strategy: Average 4 trials -> 1 sample (16,540 samples, not 66,160)
"""

import torch
import numpy as np
from pathlib import Path

def convert_to_dreamdiffusion_format(input_path, output_path, subject=1):
    """
    Convert data format by averaging trials.

    Args:
        input_path: Path to our format data (train.pt or test.pt)
        output_path: Path to save converted data
        subject: Subject ID to assign (default 1)
    """
    print(f"Loading data from {input_path}...")
    data = torch.load(input_path, weights_only=False)

    eeg = data['eeg']  # (N, 4, 63, 250) - numpy array
    labels = data['label']  # (N, 4) - numpy array
    images = data['img']  # (N, 4) - numpy array

    N, num_trials, num_channels, time_steps = eeg.shape
    print(f"  EEG shape: {eeg.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Images shape: {images.shape}")

    # Convert to tensor and average trials: (N, 4, 63, 250) -> (N, 63, 250)
    eeg_tensor = torch.as_tensor(eeg, dtype=torch.float32)
    eeg_avg = eeg_tensor.mean(dim=1)
    print(f"  EEG averaged shape: {eeg_avg.shape}")

    # Use first trial's label and image (they are the same across trials)
    labels_first = labels[:, 0]
    images_first = images[:, 0]

    # Verify consistency across trials (check first 100 samples)
    print("  Verifying consistency across trials...")
    for i in range(min(100, N)):
        assert np.all(labels[i] == labels[i, 0]), \
            f"Sample {i} has inconsistent labels across trials: {labels[i]}"
        assert np.all(images[i] == images[i, 0]), \
            f"Sample {i} has inconsistent images across trials"
    print("  ✓ Verified: labels and images are consistent across trials")

    # Build unique image list
    unique_images = []
    image_to_idx = {}
    for i in range(N):
        img_path = str(images_first[i])
        # Fix path: train_images -> training_images, test_images -> test_images (already correct)
        img_path = img_path.replace('train_images/', 'training_images/')
        if img_path not in image_to_idx:
            image_to_idx[img_path] = len(unique_images)
            unique_images.append(img_path)

    print(f"  Unique images: {len(unique_images)}")

    # Build unique label list
    unique_labels = sorted(set(labels_first.tolist()))
    print(f"  Unique labels: {len(unique_labels)}")

    # Convert to dataset format
    dataset = []
    for i in range(N):
        img_path = str(images_first[i]).replace('train_images/', 'training_images/')
        dataset.append({
            'eeg': eeg_avg[i],  # (63, 250) - averaged across 4 trials
            'label': int(labels_first[i]),
            'image': image_to_idx[img_path],
            'subject': subject
        })

    print(f"  Total dataset entries: {len(dataset)}")

    # Save in DreamDiffusion format
    output_data = {
        'dataset': dataset,
        'labels': unique_labels,
        'images': unique_images
    }

    print(f"Saving to {output_path}...")
    torch.save(output_data, output_path)
    print("Done!")

    return output_data

if __name__ == '__main__':
    # Convert train data
    print("="*80)
    print("Converting TRAIN data (averaging 4 trials per sample)")
    print("="*80)
    train_data = convert_to_dreamdiffusion_format(
        input_path='/home/yiqiuliu/DL_Project/image-eeg-data/train.pt',
        output_path='/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion.pt',
        subject=1
    )

    print("\n" + "="*80)
    print("Converting TEST data (averaging 4 trials per sample)")
    print("="*80)
    test_data = convert_to_dreamdiffusion_format(
        input_path='/home/yiqiuliu/DL_Project/image-eeg-data/test.pt',
        output_path='/home/yiqiuliu/DL_Project/image-eeg-data/test_dreamdiffusion.pt',
        subject=1
    )

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Train: {len(train_data['dataset'])} entries, {len(train_data['images'])} unique images")
    print(f"Test: {len(test_data['dataset'])} entries, {len(test_data['images'])} unique images")
    print("\nStrategy: Averaged 4 trials -> 1 sample (following VisualEEGDecoding)")
    print("\nNext steps:")
    print("1. Generate splits file for train/val split")
    print("2. Update precompute and training scripts to use new data paths")
