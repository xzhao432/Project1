import pytorch_lightning as pl
import torch


class GradientMonitor(pl.Callback):
    """Monitor gradient norms for different loss components."""

    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        """Called after loss.backward() and before optimizers do anything."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Calculate total gradient norm for the model
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log to trainer
        pl_module.log('train/grad_norm_total', total_norm, on_step=True, on_epoch=False, prog_bar=False)

        # Also print to stdout (only on rank 0 to avoid duplicate prints)
        if trainer.global_rank == 0:
            print(f"\n[Step {trainer.global_step}] Gradient norm: {total_norm:.6f}")
