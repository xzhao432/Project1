import os, sys
import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy

# own code
from config import Config_Generative_Model
from dataset_precomputed import create_precomputed_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM
from eval_metrics import get_similarity_metric
from training_logger import DetailedTrainingLogger
from gradient_monitor import GradientMonitor
from validation_visualization import FixedValidationVisualization


def wandb_init(config, output_path):
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))

    # Note: This function is called after training completes, typically on a single process
    # No distributed synchronization needed here

    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                        n_way=50, num_trials=50, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list

def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    # Use model.model.generate() which supports precomputed VAE latents
    # This correctly decodes ground truth from vae_latent_precomputed instead of using dummy images
    grid, _ = generative_model.model.generate(eeg_latents_dataset_train, config.num_samples,
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))

    grid, samples = generative_model.model.generate(eeg_latents_dataset_test, config.num_samples,
                config.ddim_steps, config.HW)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs[1:]):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path,
                            f'./test{sp_idx}-{copy_idx}.png'))

    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def create_trainer(
    num_epoch,
    precision,
    accumulate_grad,
    logger,
    output_path=None,
    validation_dataset=None,
):
    # Create detailed training logger callback
    callbacks = []
    if output_path is not None:
        detailed_logger = DetailedTrainingLogger(log_dir=output_path)
        callbacks.append(detailed_logger)

    # Add gradient monitoring callback
    gradient_monitor = GradientMonitor(log_every_n_steps=100)
    callbacks.append(gradient_monitor)

    if output_path is not None and validation_dataset is not None:
        callbacks.append(
            FixedValidationVisualization(
                dataset=validation_dataset,
                output_path=output_path,
                num_items=10,
                num_samples=2,
                ddim_steps=50,
                every_n_epochs=2,
            )
        )

    # Auto-detect number of available GPUs from CUDA_VISIBLE_DEVICES
    num_gpus = torch.cuda.device_count()
    strategy = 'ddp' if num_gpus > 1 else 'auto'

    return pl.Trainer(accelerator='gpu', devices=num_gpus, max_epochs=num_epoch,
                      logger=logger, precision=precision,
                      accumulate_grad_batches=accumulate_grad,
                      limit_val_batches=0,
                      num_sanity_val_steps=0,
                      callbacks=callbacks,
                      strategy=strategy)

def main(config):
    # project setup
    import torch.serialization
    from config import Config_Generative_Model as ConfigClass

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 50)
    print("Using PRECOMPUTED features for training")
    print("This skips VAE encoding and CLIP feature extraction")
    print("=" * 50)

    if config.dataset == 'EEG':
        # Use precomputed dataset instead of regular dataset
        eeg_latents_dataset_train, eeg_latents_dataset_test = create_precomputed_EEG_dataset(
            eeg_signals_path=config.eeg_signals_path,
            precomputed_train_path=config.precomputed_train_path,
            precomputed_test_path=config.precomputed_test_path,
            subject=config.subject
        )
        num_voxels = eeg_latents_dataset_train.data_len
    else:
        raise NotImplementedError

    # prepare pretrained mbm
    torch.serialization.add_safe_globals([ConfigClass])
    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu', weights_only=False)

    # create generative model
    generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond, clip_tune = config.clip_tune, cls_tune = config.cls_tune)

    # resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        generative_model.model.load_state_dict(model_meta['model_state_dict'])
        print('model resumed')

    # finetune the model
    trainer = create_trainer(
        config.num_epoch,
        config.precision,
        config.accumulate_grad,
        config.logger,
        output_path=config.output_path,
        validation_dataset=eeg_latents_dataset_test,
    )
    generative_model.finetune(trainer, eeg_latents_dataset_train, eeg_latents_dataset_test,
                config.batch_size, config.lr, config.output_path, config=config)

    # generate images
    generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config)

    return

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning with Precomputed Features', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str, default = '../dreamdiffusion/')
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # EEG data parameters
    parser.add_argument('--eeg_signals_path', type=str, required=True, help='Path to EEG signals .pt file')
    parser.add_argument('--subject', type=int, default=0, help='Subject number')

    # precomputed features paths
    parser.add_argument('--precomputed_train_path', type=str, required=True, help='Path to precomputed train features .h5 file')
    parser.add_argument('--precomputed_test_path', type=str, required=True, help='Path to precomputed test features .h5 file')

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=lambda x: x.lower() == 'true')
    parser.add_argument('--clip_tune', type=lambda x: x.lower() == 'true')
    parser.add_argument('--cls_tune', type=lambda x: x.lower() == 'true')

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=lambda x: x.lower() == 'true')
    parser.add_argument('--eval_avg', type=lambda x: x.lower() == 'true')

    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))

    # Add new attributes that don't exist in config
    if hasattr(args, 'precomputed_train_path') and args.precomputed_train_path is not None:
        config.precomputed_train_path = args.precomputed_train_path
    if hasattr(args, 'precomputed_test_path') and args.precomputed_test_path is not None:
        config.precomputed_test_path = args.precomputed_test_path

    return config

if __name__ == '__main__':
    config = Config_Generative_Model()
    output_path = os.path.join(config.root_path, 'results', 'generation',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    config.logger = None  # Disable wandb

    args = get_args_parser()
    args = args.parse_args()
    config = update_config(args, config)

    # wandb_init(config, output_path)
    main(config)
    # wandb_finish()
