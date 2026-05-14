import os

import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data._utils.collate import default_collate


class FixedValidationVisualization(pl.Callback):
    """Generate a fixed, lightweight validation grid from evenly spaced samples."""

    def __init__(
        self,
        dataset,
        output_path,
        num_items=10,
        num_samples=2,
        ddim_steps=50,
        every_n_epochs=2,
    ):
        super().__init__()
        if len(dataset) == 0:
            raise ValueError("Validation visualization dataset is empty.")

        self.dataset = dataset
        self.output_path = output_path
        self.num_items = min(num_items, len(dataset))
        self.num_samples = num_samples
        self.ddim_steps = ddim_steps
        self.every_n_epochs = every_n_epochs
        self.indices = self._evenly_spaced_indices(len(dataset), self.num_items)
        self.fixed_batch = None

    @staticmethod
    def _evenly_spaced_indices(dataset_len, num_items):
        if num_items <= 0:
            return []
        if num_items >= dataset_len:
            return list(range(dataset_len))
        return np.linspace(0, dataset_len - 1, num_items, dtype=int).tolist()

    def _build_fixed_batch(self):
        samples = [self.dataset[idx] for idx in self.indices]
        return default_collate(samples)

    def _should_run(self, trainer):
        epoch_num = trainer.current_epoch + 1
        return epoch_num % self.every_n_epochs == 0

    @staticmethod
    def _is_rank_zero(trainer, pl_module):
        return getattr(trainer, "global_rank", getattr(pl_module, "global_rank", 0)) == 0

    @staticmethod
    def _distributed_barrier():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        if not self._should_run(trainer):
            return

        if self._is_rank_zero(trainer, pl_module):
            if self.fixed_batch is None:
                self.fixed_batch = self._build_fixed_batch()
                print(f"\n[ValidationVisualization] Fixed sample indices: {self.indices}")

            epoch_num = trainer.current_epoch + 1
            print(
                f"\n[ValidationVisualization] Generating {self.num_items} samples "
                f"for epoch {epoch_num} with {self.ddim_steps} DDIM steps."
            )

            was_training = pl_module.training
            grid, _, _ = pl_module.generate(
                self.fixed_batch,
                num_samples=self.num_samples,
                ddim_steps=self.ddim_steps,
                HW=None,
                limit=self.num_items,
                state=None,
            )
            if was_training:
                pl_module.train()
                if hasattr(pl_module, "cond_stage_model"):
                    pl_module.cond_stage_model.train()

            os.makedirs(self.output_path, exist_ok=True)
            save_path = os.path.join(self.output_path, f"val_epoch{epoch_num:03d}.png")
            Image.fromarray(grid.astype(np.uint8)).save(save_path)
            print(f"[ValidationVisualization] Saved visualization to: {save_path}")

        self._distributed_barrier()
