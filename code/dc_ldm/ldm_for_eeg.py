import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import eeg_encoder, classify_network, mapping 
from PIL import Image


class VisualEEGConditioner(nn.Module):
    """Conditioner backed by a frozen VisualEEGDecoding retrieval encoder."""

    def __init__(
        self,
        checkpoint_path,
        cond_dim,
        channels=63,
        temporal_len=250,
        proj_dim=1024,
        num_tokens=77,
        freeze_encoder=True,
    ):
        super().__init__()
        from pathlib import Path
        import importlib.util

        encoder_path = Path('/home/yiqiuliu/VisualEEGDecoding/models/Encoder.py')
        spec = importlib.util.spec_from_file_location('visualeeg_encoder', encoder_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load VisualEEG encoder from {encoder_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.encoder = mod.Brain_Visual_Encoder_EEG(
            channels=channels,
            proj_dim=proj_dim,
            temporal_len=temporal_len,
        )
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.encoder.load_state_dict(state)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self._apply_encoder_freeze()

        self.channels = channels
        self.temporal_len = temporal_len
        self.proj_dim = proj_dim
        self.num_tokens = num_tokens
        self.fmri_seq_len = num_tokens
        self.fmri_latent_dim = proj_dim
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, 256) * 0.02)
        self.token_mapper = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 256),
            nn.GELU(),
        )
        self.token_out = nn.Linear(256, cond_dim)
        self.clip_projection = nn.Linear(proj_dim, 768)
        print(
            f"[VisualEEGConditioner] Loaded {checkpoint_path} "
            f"({channels} channels, {temporal_len} timepoints, proj_dim={proj_dim})"
        )
        if freeze_encoder:
            print("[VisualEEGConditioner] Retrieval encoder frozen.")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self._apply_encoder_freeze()
        return self

    def _apply_encoder_freeze(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def projector_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith(('token_queries', 'token_mapper.', 'token_out.')):
                yield p

    def trainable_parameters(self):
        if not self.freeze_encoder:
            yield from self.parameters()
            return

        yield from self.projector_parameters()

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected retrieval EEG [B,C,T], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} EEG channels, got {x.shape[1]}")
        if x.shape[-1] != self.temporal_len:
            x = F.interpolate(x, size=self.temporal_len, mode='linear', align_corners=False)

        if self.freeze_encoder:
            with torch.no_grad():
                retrieval_emb = self.encoder(x)
        else:
            retrieval_emb = self.encoder(x)

        token_context = self.token_mapper(retrieval_emb).unsqueeze(1)
        tokens = self.token_out(token_context + self.token_queries)
        return tokens, retrieval_emb

    def get_clip_loss(self, x, image_embeds):
        target_emb = self.clip_projection(x)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss


def create_model_from_config(config, num_voxels, global_pool):
    model = eeg_encoder(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels=440, cond_dim=1280, global_pool=True, clip_tune = True, cls_tune = False,
                 input_channels=63, pretrained_channels=128, adapter_warmup_epochs=2):
        super().__init__()
        self.input_channels = input_channels
        self.pretrained_channels = pretrained_channels
        self.adapter_warmup_epochs = adapter_warmup_epochs

        # Channel adapter: adapt input channels to pretrained model's expected channels
        if input_channels != pretrained_channels:
            self.channel_adapter = nn.Conv1d(
                in_channels=input_channels,
                out_channels=pretrained_channels,
                kernel_size=1,
                bias=True
            )
            print(f"[ChannelAdapter] Created adapter: {input_channels} -> {pretrained_channels} channels")
            print(f"[ChannelAdapter] Warmup epochs: {adapter_warmup_epochs}")
        else:
            self.channel_adapter = nn.Identity()
            print(f"[ChannelAdapter] No adapter needed: input_channels == pretrained_channels == {input_channels}")

        # prepare pretrained fmri mae
        if metafile is not None:
            model = create_model_from_config(metafile['config'], num_voxels, global_pool)

            model.load_checkpoint(metafile['model_state_dict'])
        else:
            model = eeg_encoder(time_len=num_voxels, global_pool=global_pool)
        self.mae = model

        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim

        if clip_tune:
            self.mapping = mapping(input_dim=self.fmri_latent_dim)
        if cls_tune:
            self.cls_net = classify_network()
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

        # self.image_embedder = FrozenImageEmbedder()

    # def forward(self, x):
    #     # n, c, w = x.shape
    #     latent_crossattn = self.mae(x)
    #     if self.global_pool == False:
    #         latent_crossattn = self.channel_mapper(latent_crossattn)
    #     latent_crossattn = self.dim_mapper(latent_crossattn)
    #     out = latent_crossattn
    #     return out

    def forward(self, x):
        # x: [B, C, T], e.g., [B, 63, 512]
        # Apply channel adapter first
        x = self.channel_adapter(x)  # [B, 63, 512] -> [B, 128, 512]

        latent_crossattn = self.mae(x)
        latent_return = latent_crossattn
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        # image_embeds = self.image_embedder(image_inputs)
        target_emb = self.mapping(x)
        # similarity_matrix = nn.functional.cosine_similarity(target_emb.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
        # loss = clip_loss(similarity_matrix)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss
    


class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune = True, cls_tune = False,
                 eeg_input_channels=63, eeg_pretrained_channels=128, adapter_warmup_epochs=2,
                 use_visual_eeg_encoder=False, visual_eeg_checkpoint_path=None,
                 visual_eeg_channels=63, visual_eeg_temporal_len=250,
                 visual_eeg_proj_dim=1024, freeze_visual_eeg_encoder=True,
                 visual_eeg_projector_only=True):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.ckp_path = os.path.join(pretrain_root, 'models/eeg_pretrain_scp/v1-5-pruned.ckpt')
        self.config_path = os.path.join(pretrain_root, 'models/config15.yaml') 
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim
        if use_visual_eeg_encoder:
            # The default CLIP text conditioner is immediately replaced below.
            # Use a cheap placeholder to avoid loading unused CLIP weights.
            config.model.params.cond_stage_config = OmegaConf.create({'target': 'torch.nn.Identity'})

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu", weights_only=False)['state_dict']

        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        if use_visual_eeg_encoder:
            if clip_tune:
                print("[VisualEEG] clip_tune=True ignored; disabling side-projection CLIP loss for VisualEEG conditioner.")
                clip_tune = False
            if visual_eeg_checkpoint_path is None:
                raise ValueError("visual_eeg_checkpoint_path is required when use_visual_eeg_encoder=True")
            model.cond_stage_model = VisualEEGConditioner(
                checkpoint_path=visual_eeg_checkpoint_path,
                cond_dim=self.cond_dim,
                channels=visual_eeg_channels,
                temporal_len=visual_eeg_temporal_len,
                proj_dim=visual_eeg_proj_dim,
                freeze_encoder=freeze_visual_eeg_encoder,
            )
            model.use_visual_eeg_encoder = True
            model.visual_eeg_projector_only = visual_eeg_projector_only
        else:
            model.cond_stage_model = cond_stage_model(
                metafile, num_voxels, self.cond_dim,
                global_pool=global_pool,
                clip_tune=clip_tune,
                cls_tune=cls_tune,
                input_channels=eeg_input_channels,
                pretrained_channels=eeg_pretrained_channels,
                adapter_warmup_epochs=adapter_warmup_epochs
            )
            model.use_visual_eeg_encoder = False
            model.visual_eeg_projector_only = False

        model.ddim_steps = ddim_steps
        if getattr(model, 'use_visual_eeg_encoder', False) and getattr(model, 'visual_eeg_projector_only', False):
            model.use_ema = False
            if hasattr(model, 'model_ema'):
                delattr(model, 'model_ema')
            print("[VisualEEG] Projector-only mode: freezing UNet/cross-attn/VAE; training conditioner projector only.")
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=bs1, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)
        if getattr(self.model, 'visual_eeg_projector_only', False):
            self.model.apply_visual_eeg_projector_only_freeze()
        else:
            self.model.unfreeze_whole_model()
            if hasattr(self.model.cond_stage_model, '_apply_encoder_freeze'):
                self.model.cond_stage_model._apply_encoder_freeze()
            self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        if getattr(self.model, 'visual_eeg_projector_only', False):
            self.model.apply_visual_eeg_projector_only_freeze()
        else:
            self.model.unfreeze_whole_model()
            if hasattr(self.model.cond_stage_model, '_apply_encoder_freeze'):
                self.model.cond_stage_model._apply_encoder_freeze()
        
        if getattr(trainers, 'global_rank', 0) == 0:
            torch.save(
                {
                    'model_state_dict': self.model.state_dict(),
                    'config': config,
                    'state': torch.random.get_rng_state()

                },
                os.path.join(output_path, 'checkpoint.pth')
            )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                # GT image is already in [0, 1] range, no need to transform
                gt_image = torch.clamp(gt_image, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)




class eLDM_eval:

    def __init__(self, config_path, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune = True, cls_tune = False):
        self.config_path = config_path # 
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)

        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(None, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune = clip_tune,cls_tune = cls_tune)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=bs1, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        
        if getattr(trainers, 'global_rank', 0) == 0:
            torch.save(
                {
                    'model_state_dict': self.model.state_dict(),
                    'config': config,
                    'state': torch.random.get_rng_state()

                },
                os.path.join(output_path, 'checkpoint.pth')
            )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                # print(item)
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                # GT image is already in [0, 1] range, no need to transform
                gt_image = torch.clamp(gt_image, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
