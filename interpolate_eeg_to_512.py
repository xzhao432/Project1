"""
Interpolate EEG data from 250 time points to 512 time points.
This is needed for compatibility with DreamDiffusion's pretrained encoder.
"""

import torch
import torch.nn.functional as F
from pathlib import Path

def interpolate_eeg(eeg_tensor, target_length=512):
    """
    Interpolate EEG from current length to target length.

    Args:
        eeg_tensor: (channels, time) tensor
        target_length: target time dimension

    Returns:
        Interpolated tensor of shape (channels, target_length)
    """
    # Add batch dimension: (channels, time) -> (1, channels, time)
    eeg = eeg_tensor.unsqueeze(0)

    # Interpolate along time dimension
    # Input: (batch, channels, time)
    # Output: (batch, channels, target_length)
    eeg_interp = F.interpolate(
        eeg,
        size=target_length,
        mode='linear',
        align_corners=True
    )

    # Remove batch dimension: (1, channels, target_length) -> (channels, target_length)
    return eeg_interp.squeeze(0)


def process_dataset(input_path, output_path, target_length=512):
    """Process entire dataset and save interpolated version."""
    print(f"Loading data from {input_path}")
    data = torch.load(input_path, weights_only=False)

    print(f"Original dataset: {len(data['dataset'])} samples")

    # Check original shape
    sample_eeg = data['dataset'][0]['eeg']
    print(f"Original EEG shape: {sample_eeg.shape}")

    # Interpolate all EEG data
    print(f"Interpolating to {target_length} time points...")
    for i, sample in enumerate(data['dataset']):
        original_eeg = sample['eeg']
        interpolated_eeg = interpolate_eeg(original_eeg, target_length)
        sample['eeg'] = interpolated_eeg

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data['dataset'])} samples")

    # Verify
    sample_eeg_new = data['dataset'][0]['eeg']
    print(f"New EEG shape: {sample_eeg_new.shape}")

    # Fix image paths (train_images -> training_images)
    print("Fixing image paths...")
    data['images'] = [img.replace('train_images/', 'training_images/') for img in data['images']]

    # Save
    print(f"Saving to {output_path}")
    torch.save(data, output_path)
    print("Done!")


if __name__ == "__main__":
    # Process train data
    train_input = "/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion.pt"
    train_output = "/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion_512.pt"

    print("="*60)
    print("Processing training data")
    print("="*60)
    process_dataset(train_input, train_output, target_length=512)

    print("\n" + "="*60)
    print("Processing test data")
    print("="*60)

    # Process test data
    test_input = "/home/yiqiuliu/DL_Project/image-eeg-data/test_dreamdiffusion.pt"
    test_output = "/home/yiqiuliu/DL_Project/image-eeg-data/test_dreamdiffusion_512.pt"

    process_dataset(test_input, test_output, target_length=512)

    print("\n" + "="*60)
    print("All done! New files created:")
    print(f"  {train_output}")
    print(f"  {test_output}")
    print("="*60)
