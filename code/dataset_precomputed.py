"""
Dataset class for loading precomputed features from HDF5 files.
This skips VAE encoding and CLIP feature extraction during training.
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

class PrecomputedEEGDataset(Dataset):
    """
    Dataset that loads precomputed VAE latents and CLIP features from HDF5 files.
    This significantly speeds up training by skipping expensive VAE encoding and CLIP extraction.
    """

    def __init__(self, eeg_signals_path, precomputed_h5_path, subject=0, retrieval_eeg_signals_path=None):
        """
        Args:
            eeg_signals_path: Path to EEG signals .pt file
            precomputed_h5_path: Path to precomputed features .h5 file
            subject: Subject number (default: 0)
            retrieval_eeg_signals_path: Optional path to 63x250 EEG tensors for
                the VisualEEGDecoding retrieval encoder.
        """
        self.subject = subject

        # Load EEG data
        loaded = torch.load(eeg_signals_path, weights_only=False)
        if subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if loaded['dataset'][i]['subject'] == subject]
        else:
            self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.num_voxels = 440
        self.data_len = 512

        self.retrieval_data = None
        self.retrieval_data_len = 250
        if retrieval_eeg_signals_path is not None:
            retrieval_loaded = torch.load(retrieval_eeg_signals_path, weights_only=False)
            if subject != 0:
                self.retrieval_data = [
                    retrieval_loaded['dataset'][i]
                    for i in range(len(retrieval_loaded['dataset']))
                    if retrieval_loaded['dataset'][i]['subject'] == subject
                ]
            else:
                self.retrieval_data = retrieval_loaded['dataset']

        # Open HDF5 file (keep it open for fast access)
        self.h5_file = h5py.File(precomputed_h5_path, 'r')

        # Get indices from HDF5
        self.indices = self.h5_file['indices'][:]
        self.precomputed_len = len(self.indices)

        print(f"Loaded precomputed dataset: {self.precomputed_len} samples")
        print(f"  Original EEG data: {len(self.data)} samples")
        if self.retrieval_data is not None:
            print(f"  Retrieval EEG data: {len(self.retrieval_data)} samples")
        print(f"  VAE latents shape: {self.h5_file['vae_latents'].shape}")
        print(f"  CLIP features shape: {self.h5_file['clip_features'].shape}")

    def __len__(self):
        return self.precomputed_len

    def __getitem__(self, idx):
        # Get the original data index (convert to Python int for dict access)
        original_idx = int(self.indices[idx])

        # Validate index bounds
        if original_idx >= len(self.data):
            raise IndexError(f"Precomputed index {original_idx} out of bounds for EEG data (size: {len(self.data)}). "
                           f"The precomputed features file may be out of sync with the EEG data file.")

        # Get EEG data from the list and apply same preprocessing as original dataset
        eeg = self.data[original_idx]["eeg"].float().t()

        # Slice to 440 time points (20:460)
        eeg = eeg[20:460, :]

        # Interpolate from 440 to 512 time points
        eeg = np.array(eeg.transpose(0, 1))
        x = np.linspace(0, 1, eeg.shape[-1])  # 440 points
        x2 = np.linspace(0, 1, self.data_len)  # 512 points
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()

        eeg_retrieval = None
        if self.retrieval_data is not None:
            if original_idx >= len(self.retrieval_data):
                raise IndexError(
                    f"Precomputed index {original_idx} out of bounds for retrieval EEG data "
                    f"(size: {len(self.retrieval_data)})."
                )
            eeg_retrieval = self.retrieval_data[original_idx]["eeg"].float()
            if eeg_retrieval.ndim != 2:
                raise ValueError(f"Expected retrieval EEG [C,T], got {tuple(eeg_retrieval.shape)}")
            if eeg_retrieval.shape[-1] != self.retrieval_data_len:
                eeg_np = eeg_retrieval.numpy()
                x = np.linspace(0, 1, eeg_np.shape[-1])
                x2 = np.linspace(0, 1, self.retrieval_data_len)
                eeg_retrieval = torch.from_numpy(interp1d(x, eeg_np)(x2)).float()

        # Get label
        label = torch.tensor(self.data[original_idx]["label"]).long()

        # Load precomputed features from HDF5
        vae_latent = torch.from_numpy(self.h5_file['vae_latents'][idx])
        clip_feature = torch.from_numpy(self.h5_file['clip_features'][idx])

        # For compatibility, we still need to return 'image' and 'image_raw'
        # but they will be replaced by precomputed latents in the model
        # Use minimal dummy tensors to save memory
        dummy_image = torch.zeros(1, 1, 1)  # Minimal placeholder
        dummy_image_raw = {'pixel_values': torch.zeros(1, 1, 1)}  # Minimal placeholder

        sample = {
            'eeg': eeg,
            'label': label,
            'image': dummy_image,  # Placeholder
            'image_raw': dummy_image_raw,  # Placeholder
            'vae_latent_precomputed': vae_latent,  # Precomputed VAE latent
            'clip_feature_precomputed': clip_feature,  # Precomputed CLIP feature
        }
        if eeg_retrieval is not None:
            sample['eeg_retrieval'] = eeg_retrieval
        return sample

    def __del__(self):
        # Close HDF5 file when dataset is destroyed
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass  # Silently ignore errors during cleanup


def create_precomputed_EEG_dataset(
    eeg_signals_path,
    precomputed_train_path,
    precomputed_test_path,
    subject=0,
    retrieval_eeg_signals_path=None,
):
    """
    Create train and test datasets using precomputed features.

    Args:
        eeg_signals_path: Path to EEG signals .pt file
        precomputed_train_path: Path to train precomputed .h5 file
        precomputed_test_path: Path to test precomputed .h5 file
        subject: Subject number
        retrieval_eeg_signals_path: Optional path to 63x250 EEG tensors for
            the VisualEEGDecoding retrieval encoder.

    Returns:
        dataset_train, dataset_test
    """
    dataset_train = PrecomputedEEGDataset(
        eeg_signals_path,
        precomputed_train_path,
        subject,
        retrieval_eeg_signals_path=retrieval_eeg_signals_path,
    )
    dataset_test = PrecomputedEEGDataset(
        eeg_signals_path,
        precomputed_test_path,
        subject,
        retrieval_eeg_signals_path=retrieval_eeg_signals_path,
    )

    return dataset_train, dataset_test
