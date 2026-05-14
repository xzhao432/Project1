"""
Custom training logger for detailed epoch-by-epoch metrics logging.
"""
import logging
import os
import time
from datetime import datetime
from pytorch_lightning.callbacks import Callback


class DetailedTrainingLogger(Callback):
    """
    Custom callback to log detailed training and validation metrics to a file.
    Similar to BasicTS logging format.
    """

    def __init__(self, log_dir, log_name=None):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            log_name = f"training_log_{timestamp}.log"

        log_path = os.path.join(log_dir, log_name)

        # Setup logger
        self.logger = logging.getLogger('DreamDiffusion-training')
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Timing
        self.epoch_start_time = None
        self.train_start_time = None

        self.logger.info(f"Training log saved to: {log_path}")

    def on_train_start(self, trainer, pl_module):
        self.logger.info("Initializing training.")
        self.logger.info(f"Total epochs: {trainer.max_epochs}")

        # Count parameters
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params}")
        self.logger.info(f"Trainable parameters: {trainable_params}")

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        self.logger.info(f"Epoch {current_epoch} / {max_epochs}")

    def on_train_epoch_end(self, trainer, pl_module):
        # Get training metrics
        train_time = time.time() - self.epoch_start_time

        # Collect metrics from logged values
        metrics = trainer.callback_metrics

        # Format train metrics
        train_metrics = []
        train_metrics.append(f"train/time: {train_time:.2f} (s)")

        for key, value in metrics.items():
            if key.startswith('train/'):
                metric_name = key.replace('train/', '')
                train_metrics.append(f"train/{metric_name}: {value:.4f}")

        if train_metrics:
            metrics_str = ", ".join(train_metrics)
            self.logger.info(f"Result <train>: [{metrics_str}]")

    def on_validation_start(self, trainer, pl_module):
        if trainer.current_epoch > 0:  # Skip first validation (sanity check)
            self.logger.info("Start validation.")
        self.val_start_time = time.time()

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch == 0:  # Skip logging for sanity check
            return

        val_time = time.time() - self.val_start_time

        # Collect validation metrics
        metrics = trainer.callback_metrics

        # Format val metrics
        val_metrics = []
        val_metrics.append(f"val/time: {val_time:.2f} (s)")

        for key, value in metrics.items():
            if key.startswith('val/'):
                metric_name = key.replace('val/', '')
                val_metrics.append(f"val/{metric_name}: {value:.4f}")

        if val_metrics:
            metrics_str = ", ".join(val_metrics)
            self.logger.info(f"Result <val>: [{metrics_str}]")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        current_epoch = trainer.current_epoch + 1
        self.logger.info(f"Checkpoint saved at epoch {current_epoch}")

    def on_train_end(self, trainer, pl_module):
        self.logger.info("Training completed!")
