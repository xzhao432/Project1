import os

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data._utils.collate import default_collate


class FixedConditioningProbe(pl.Callback):
    """Track whether EEG-conditioned tokens improve on a fixed denoising probe."""

    def __init__(
        self,
        dataset,
        output_path,
        num_items=32,
        every_n_epochs=1,
        timesteps=(50, 250, 500, 750),
        seed=2026,
    ):
        super().__init__()
        if len(dataset) == 0:
            raise ValueError("Conditioning probe dataset is empty.")

        self.dataset = dataset
        self.output_path = output_path
        self.num_items = min(num_items, len(dataset))
        if self.num_items <= 0:
            raise ValueError("Conditioning probe requires at least one sample.")
        self.every_n_epochs = every_n_epochs
        self.timesteps = tuple(timesteps)
        self.seed = seed
        self.indices = self._evenly_spaced_indices(len(dataset), self.num_items)
        self.fixed_batch = None
        self.fixed_noise = None
        self.fixed_timesteps = None

    @staticmethod
    def _evenly_spaced_indices(dataset_len, num_items):
        if num_items <= 0:
            return []
        if num_items >= dataset_len:
            return list(range(dataset_len))
        return np.linspace(0, dataset_len - 1, num_items, dtype=int).tolist()

    @staticmethod
    def _is_rank_zero(trainer, pl_module):
        return getattr(trainer, "global_rank", getattr(pl_module, "global_rank", 0)) == 0

    @staticmethod
    def _distributed_barrier():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _build_fixed_batch(self):
        samples = [self.dataset[idx] for idx in self.indices]
        return default_collate(samples)

    def _to_device(self, value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {k: self._to_device(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_device(v, device) for v in value]
        return value

    def _prepare_fixed_tensors(self, pl_module):
        if self.fixed_batch is None:
            self.fixed_batch = self._build_fixed_batch()
            print(f"\n[ConditioningProbe] Fixed sample indices: {self.indices}")

        device = pl_module.device
        batch = self._to_device(self.fixed_batch, device)
        z = batch["vae_latent_precomputed"].float()
        bsz = z.shape[0]

        if self.fixed_timesteps is None:
            timestep_values = torch.tensor(self.timesteps, dtype=torch.long)
            repeats = (bsz + len(timestep_values) - 1) // len(timestep_values)
            self.fixed_timesteps = timestep_values.repeat(repeats)[:bsz]
        t = self.fixed_timesteps.to(device)

        if self.fixed_noise is None or tuple(self.fixed_noise.shape) != tuple(z.shape):
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self.fixed_noise = torch.randn(z.shape, generator=generator)
        noise = self.fixed_noise.to(device)

        return batch, z, t, noise

    def _condition_from_batch(self, pl_module, batch, mode):
        eeg_key = "eeg_retrieval" if getattr(pl_module, "use_visual_eeg_encoder", False) else "eeg"
        eeg = batch[eeg_key]
        if mode == "shuffle":
            eeg = torch.roll(eeg, shifts=1, dims=0)
        elif mode == "zero":
            eeg = torch.zeros_like(eeg)
        elif mode != "correct":
            raise ValueError(f"Unknown probe mode: {mode}")
        cond, _ = pl_module.get_learned_conditioning(eeg)
        return cond

    @torch.no_grad()
    def _denoising_loss(self, pl_module, z, cond, t, noise):
        x_noisy = pl_module.q_sample(x_start=z, t=t, noise=noise)
        model_output = pl_module.apply_model(x_noisy, t, cond)
        target = z if pl_module.parameterization == "x0" else noise
        return pl_module.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_num = trainer.current_epoch + 1
        if epoch_num % self.every_n_epochs != 0:
            return

        if self._is_rank_zero(trainer, pl_module):
            was_training = pl_module.training
            pl_module.eval()
            batch, z, t, noise = self._prepare_fixed_tensors(pl_module)

            losses = {}
            for mode in ("correct", "shuffle", "zero"):
                cond = self._condition_from_batch(pl_module, batch, mode)
                per_sample = self._denoising_loss(pl_module, z, cond, t, noise)
                losses[mode] = per_sample

            correct = losses["correct"].mean()
            shuffle = losses["shuffle"].mean()
            zero = losses["zero"].mean()
            margin_shuffle = shuffle - correct
            margin_zero = zero - correct

            metrics = {
                "probe/loss_correct": correct,
                "probe/loss_shuffle": shuffle,
                "probe/loss_zero": zero,
                "probe/margin_shuffle": margin_shuffle,
                "probe/margin_zero": margin_zero,
            }
            pl_module.log_dict(
                metrics,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                rank_zero_only=True,
            )

            line = (
                f"epoch={epoch_num}, "
                f"loss_correct={correct.item():.6f}, "
                f"loss_shuffle={shuffle.item():.6f}, "
                f"loss_zero={zero.item():.6f}, "
                f"margin_shuffle={margin_shuffle.item():.6f}, "
                f"margin_zero={margin_zero.item():.6f}"
            )
            print(f"\n[ConditioningProbe] {line}")
            if self.output_path is not None:
                os.makedirs(self.output_path, exist_ok=True)
                with open(os.path.join(self.output_path, "conditioning_probe.csv"), "a") as f:
                    if f.tell() == 0:
                        f.write(
                            "epoch,loss_correct,loss_shuffle,loss_zero,"
                            "margin_shuffle,margin_zero\n"
                        )
                    f.write(
                        f"{epoch_num},{correct.item():.8f},{shuffle.item():.8f},"
                        f"{zero.item():.8f},{margin_shuffle.item():.8f},"
                        f"{margin_zero.item():.8f}\n"
                    )

            if was_training:
                pl_module.train()
                if getattr(pl_module, "visual_eeg_projector_only", False):
                    pl_module.apply_visual_eeg_projector_only_freeze()

        self._distributed_barrier()
