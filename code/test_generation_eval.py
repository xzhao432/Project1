#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from skimage.metrics import structural_similarity
import torchvision.transforms as transforms

from config import Config_Generative_Model as ConfigClass
from dc_ldm.ldm_for_eeg import eLDM
from dc_ldm.models.diffusion.ddim import DDIMSampler


BLUR_LEVELS = ["1", "3", "9", "15", "21", "27", "33", "39", "45", "51", "57", "63"]
DEFAULT_ENCODER_CHECKPOINT = (
    Path("/home/yiqiuliu/VisualEEGDecoding/runs")
    / ("ablation_" + "ret" + "rieval")
    / "channels-all_wd-0.0_temp-0.07_seed-2027"
    / "best.pth"
)


def evenly_spaced_indices(n, k):
    if k >= n:
        return list(range(n))
    return np.linspace(0, n - 1, k, dtype=int).tolist()


def load_h5_arrays(path):
    with h5py.File(path, "r") as f:
        return {
            "indices": f["indices"][:],
            "vae_latents": torch.from_numpy(f["vae_latents"][:]).float(),
        }


def load_visual_test200(path, image_root):
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    images = loaded["images"]
    return {
        "indices": np.arange(len(loaded["dataset"]), dtype=np.int32),
        "vae_latents": None,
        "dataset": loaded["dataset"],
        "images": images,
        "image_paths": [Path(image_root) / img for img in images],
    }


def load_gallery_eeg(path):
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return loaded["dataset"], loaded["images"]


def load_multiblur_features(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    keys = sorted(cache["1"].keys())
    stem_to_feature = {}
    for key in keys:
        feats = torch.stack([cache[level][key].float() for level in BLUR_LEVELS], dim=0)
        stem_to_feature[Path(key).stem] = F.normalize(feats, dim=-1)
    return stem_to_feature


def split_defaults(split):
    if split == "train":
        return {
            "h5": "/home/yiqiuliu/DreamDiffusion/precomputed_features/train_precomputed.h5",
            "features": "/home/yiqiuliu/VisualEEGDecoding/data/things-eeg/Image_feature/MultiBlur_RN50_train.pt",
        }
    if split == "test":
        # DreamDiffusion's test_precomputed.h5 is the held-out validation
        # subset of train_dreamdiffusion.pt, not VisualEEG's 200-image test set.
        return {
            "h5": "/home/yiqiuliu/DreamDiffusion/precomputed_features/test_precomputed.h5",
            "features": "/home/yiqiuliu/VisualEEGDecoding/data/things-eeg/Image_feature/MultiBlur_RN50_train.pt",
        }
    if split == "visual_test200":
        return {
            "pt": "/home/yiqiuliu/DL_Project/image-eeg-data/test_dreamdiffusion.pt",
            "image_root": "/home/yiqiuliu/DL_Project/image-eeg-data",
            "features": "/home/yiqiuliu/VisualEEGDecoding/data/things-eeg/Image_feature/MultiBlur_RN50_test.pt",
        }
    raise ValueError(f"Unknown split: {split}")


def image_stem(images, original_idx):
    return Path(str(images[int(original_idx)]).replace("\\", "/")).stem


def get_eeg(dataset, original_idx):
    return dataset[int(original_idx)]["eeg"].float()


def load_split_data(split, h5_override=None, pt_override=None, image_root_override=None):
    defaults = split_defaults(split)
    if split == "visual_test200":
        return load_visual_test200(
            pt_override or defaults["pt"],
            image_root_override or defaults["image_root"],
        )
    return load_h5_arrays(h5_override or defaults["h5"])


def load_visual_eeg_encoder(checkpoint_path, channels, proj_dim, temporal_len, device):
    encoder_path = Path("/home/yiqiuliu/VisualEEGDecoding/models/Encoder.py")
    spec = importlib.util.spec_from_file_location("visualeeg_encoder", encoder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load VisualEEG encoder from {encoder_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    encoder = mod.Brain_Visual_Encoder_EEG(
        channels=channels,
        proj_dim=proj_dim,
        temporal_len=temporal_len,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


@torch.no_grad()
def build_candidate_gallery(
    bank_h5,
    gallery_data,
    images,
    feature_map,
    encoder,
    device,
    batch_size=256,
    max_source_items=None,
):
    bank_indices = bank_h5["indices"]
    bank_latents = bank_h5.get("vae_latents")
    if max_source_items is not None:
        bank_indices = bank_indices[:max_source_items]
        if bank_latents is not None:
            bank_latents = bank_latents[:max_source_items]
    eeg_features = []
    img_features = []
    valid_rows = []
    stems = []

    for start in range(0, len(bank_indices), batch_size):
        end = min(start + batch_size, len(bank_indices))
        eeg_batch = []
        img_batch = []
        row_ids = []
        row_stems = []
        for row, original_idx in enumerate(bank_indices[start:end], start=start):
            stem = image_stem(images, original_idx)
            if stem not in feature_map:
                continue
            eeg_batch.append(get_eeg(gallery_data, original_idx))
            img_batch.append(feature_map[stem])
            row_ids.append(row)
            row_stems.append(stem)
        if not eeg_batch:
            continue
        eeg_tensor = torch.stack(eeg_batch, dim=0).to(device)
        img_tensor = torch.stack(img_batch, dim=0).to(device)
        ze = F.normalize(encoder(eeg_tensor), dim=-1)
        zi = F.normalize(encoder.get_image_feature(img_tensor), dim=-1)
        eeg_features.append(ze.cpu())
        img_features.append(zi.cpu())
        valid_rows.extend(row_ids)
        stems.extend(row_stems)

    if not valid_rows:
        raise RuntimeError("No source-gallery items matched the provided image feature cache.")

    return {
        "indices": bank_indices[np.array(valid_rows, dtype=np.int64)],
        "latents": bank_latents[valid_rows] if bank_latents is not None else None,
        "rows": np.array(valid_rows, dtype=np.int64),
        "eeg_features": torch.cat(eeg_features, dim=0),
        "img_features": torch.cat(img_features, dim=0),
        "stems": stems,
    }


def load_ldm(args, device):
    import torch.serialization

    torch.serialization.add_safe_globals([ConfigClass])
    metafile = torch.load(args.pretrain_mbm_path, map_location="cpu", weights_only=False)
    model_wrap = eLDM(
        metafile,
        num_voxels=512,
        device=device,
        pretrain_root=args.pretrain_gm_path,
        logger=None,
        ddim_steps=args.ddim_steps,
        global_pool=False,
        use_time_cond=True,
        clip_tune=False,
        cls_tune=False,
        use_visual_eeg_encoder=True,
        visual_eeg_checkpoint_path=args.encoder_checkpoint_path,
        visual_eeg_channels=63,
        visual_eeg_temporal_len=250,
        visual_eeg_proj_dim=1024,
        freeze_visual_eeg_encoder=True,
        visual_eeg_projector_only=True,
    )
    model = model_wrap.model
    if args.checkpoint_path:
        meta = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(meta["model_state_dict"], strict=True)
        print(f"[model] loaded checkpoint: {args.checkpoint_path}")
    model.to(device)
    model.eval()
    model.apply_visual_eeg_projector_only_freeze()
    return model


def decode_latent(model, z):
    img = model.decode_first_stage(z)
    return torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)


def tensor_to_uint8_hwc(img):
    if img.ndim == 4:
        img = img[0]
    array = rearrange(img.detach().cpu(), "c h w -> h w c").numpy()
    return (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def compute_ssim(gt_uint8, pred_uint8):
    return float(structural_similarity(gt_uint8, pred_uint8, data_range=255, channel_axis=-1))


def compute_mse(gt_uint8, pred_uint8):
    gt = gt_uint8.astype(np.float32) / 255.0
    pred = pred_uint8.astype(np.float32) / 255.0
    return float(np.square(gt - pred).mean())


def _tensor_chw(img):
    if img.ndim == 4:
        img = img[0]
    return img.detach().cpu().float().clamp(0.0, 1.0)


def compute_eval_pixcorr(gt_img, pred_img):
    resize = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    gt = resize(_tensor_chw(gt_img)).reshape(-1).numpy()
    pred = resize(_tensor_chw(pred_img)).reshape(-1).numpy()
    if np.std(gt) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(gt, pred)[0, 1])


def compute_eval_ssim(gt_img, pred_img):
    from skimage.color import rgb2gray

    resize = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    gt = rearrange(resize(_tensor_chw(gt_img)), "c h w -> h w c").numpy()
    pred = rearrange(resize(_tensor_chw(pred_img)), "c h w -> h w c").numpy()
    gt_gray = rgb2gray(gt)
    pred_gray = rgb2gray(pred)
    return float(
        structural_similarity(
            pred_gray,
            gt_gray,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
    )


class ClipImageSimilarity:
    def __init__(self, model_path, device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    @torch.no_grad()
    def score_against_gt(self, gt_uint8, pred_uint8_list):
        images = [Image.fromarray(gt_uint8)] + [Image.fromarray(pred) for pred in pred_uint8_list]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_image_features(**inputs)
        features = F.normalize(features, dim=-1)
        scores = (features[0:1] * features[1:]).sum(dim=-1)
        return [float(score) for score in scores.detach().cpu()]


def summarize_metric_records(records):
    summary = {}
    groups = {"all": records}
    variants = sorted({r["variant"] for r in records})
    metrics = ["ssim", "mse", "clip_similarity", "eval_pixcorr", "eval_ssim"]

    for group_name, group_records in groups.items():
        summary[group_name] = {}
        for variant in variants:
            subset = [r for r in group_records if r["variant"] == variant]
            summary[group_name][variant] = {"count": len(subset)}
            for metric in metrics:
                values = np.array([r[metric] for r in subset], dtype=np.float64)
                if len(values) == 0:
                    summary[group_name][variant][metric] = None
                    continue
                summary[group_name][variant][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
    return summary


def _two_way_identification(real_features, pred_features):
    real_features = real_features.float().flatten(1).detach().cpu().numpy()
    pred_features = pred_features.float().flatten(1).detach().cpu().numpy()
    n = len(real_features)
    if n <= 1:
        return float("nan")
    corr = np.corrcoef(real_features, pred_features)
    corr = corr[:n, n:]
    congruent = np.diag(corr)
    success = corr < congruent
    success_count = np.sum(success, axis=0)
    return float(np.mean(success_count) / (n - 1))


@torch.no_grad()
def _extract_feature_map(images, model, preprocess, feature_key, device):
    batch = preprocess(images.to(device).float())
    features = model(batch)
    if feature_key is None:
        return features.float().flatten(1)
    return features[feature_key].float().flatten(1)


def _safe_official_metric(metrics, key, fn):
    try:
        metrics[key] = float(fn())
    except Exception as exc:
        metrics[key] = None
        print(f"[warn] skipped {key}: {exc}")


@torch.no_grad()
def compute_official_identification_metrics(gt_images, variant_images, clip_model_path, device, include_swav=False):
    from torchvision.models import AlexNet_Weights, Inception_V3_Weights, alexnet, inception_v3
    from torchvision.models.feature_extraction import create_feature_extractor

    gt_images = gt_images.to(device).float()
    summary = {}

    for variant_name, pred_images in variant_images.items():
        pred_images = pred_images.to(device).float()
        variant_metrics = {"count": int(len(pred_images))}

        def alex_metrics():
            weights = AlexNet_Weights.IMAGENET1K_V1
            model = create_feature_extractor(
                alexnet(weights=weights), return_nodes=["features.4", "features.11"]
            ).to(device)
            model.eval().requires_grad_(False)
            preprocess = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            real2 = _extract_feature_map(gt_images, model, preprocess, "features.4", device)
            pred2 = _extract_feature_map(pred_images, model, preprocess, "features.4", device)
            real5 = _extract_feature_map(gt_images, model, preprocess, "features.11", device)
            pred5 = _extract_feature_map(pred_images, model, preprocess, "features.11", device)
            return _two_way_identification(real2, pred2), _two_way_identification(real5, pred5)

        try:
            alex2, alex5 = alex_metrics()
            variant_metrics["eval_alex2"] = alex2
            variant_metrics["eval_alex5"] = alex5
        except Exception as exc:
            variant_metrics["eval_alex2"] = None
            variant_metrics["eval_alex5"] = None
            print(f"[warn] skipped {variant_name} AlexNet metrics: {exc}")

        def inception_metric():
            weights = Inception_V3_Weights.DEFAULT
            model = create_feature_extractor(
                inception_v3(weights=weights), return_nodes=["avgpool"]
            ).to(device)
            model.eval().requires_grad_(False)
            preprocess = transforms.Compose([
                transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            real = _extract_feature_map(gt_images, model, preprocess, "avgpool", device)
            pred = _extract_feature_map(pred_images, model, preprocess, "avgpool", device)
            return _two_way_identification(real, pred)

        def clip_identification_metric():
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
            processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
            model.eval()

            def features_from_tensor(images):
                pil_images = [Image.fromarray(tensor_to_uint8_hwc(image)) for image in images.detach().cpu()]
                inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
                return model.get_image_features(**inputs).float().flatten(1)

            real = features_from_tensor(gt_images)
            pred = features_from_tensor(pred_images)
            return _two_way_identification(real, pred)

        def effnet_metric():
            import scipy as sp
            from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

            weights = EfficientNet_B1_Weights.DEFAULT
            model = create_feature_extractor(
                efficientnet_b1(weights=weights), return_nodes=["avgpool"]
            ).to(device)
            model.eval().requires_grad_(False)
            preprocess = transforms.Compose([
                transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            real = _extract_feature_map(gt_images, model, preprocess, "avgpool", device).cpu().numpy()
            pred = _extract_feature_map(pred_images, model, preprocess, "avgpool", device).cpu().numpy()
            distances = [sp.spatial.distance.correlation(real[i], pred[i]) for i in range(len(real))]
            return float(np.mean(distances))

        _safe_official_metric(variant_metrics, "eval_inception", inception_metric)
        _safe_official_metric(variant_metrics, "eval_clip", clip_identification_metric)
        _safe_official_metric(variant_metrics, "eval_effnet", effnet_metric)

        if include_swav:
            def swav_metric():
                import scipy as sp

                swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
                model = create_feature_extractor(swav_model, return_nodes=["avgpool"]).to(device)
                model.eval().requires_grad_(False)
                preprocess = transforms.Compose([
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                real = _extract_feature_map(gt_images, model, preprocess, "avgpool", device).cpu().numpy()
                pred = _extract_feature_map(pred_images, model, preprocess, "avgpool", device).cpu().numpy()
                distances = [sp.spatial.distance.correlation(real[i], pred[i]) for i in range(len(real))]
                return float(np.mean(distances))

            _safe_official_metric(variant_metrics, "eval_swav", swav_metric)

        summary[variant_name] = variant_metrics

    return summary


def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    img = transforms.Resize((512, 512))(img)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    tensor = rearrange(tensor, "h w c -> c h w")
    return tensor * 2.0 - 1.0


@torch.no_grad()
def encode_image_latents(model, image_paths, device, batch_size=8):
    latents = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch = torch.stack([load_image_tensor(path) for path in batch_paths], dim=0).to(device)
        posterior = model.encode_first_stage(batch)
        latents.append(model.get_first_stage_encoding(posterior).cpu())
    return torch.cat(latents, dim=0)


@torch.no_grad()
def ddim_latent_refinement(model, sampler, init_latent, cond, strength, ddim_steps, noise):
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
    t_enc = max(1, min(ddim_steps, int(round(strength * ddim_steps))))
    step = int(sampler.ddim_timesteps[t_enc - 1])
    t = torch.full((init_latent.shape[0],), step, device=init_latent.device, dtype=torch.long)
    x_t = model.q_sample(x_start=init_latent, t=t, noise=noise)
    samples, _ = sampler.ddim_sampling(
        cond,
        x_t.shape,
        x_T=x_t,
        # Existing DDIMSampler keeps the first ``timesteps - 1`` DDIM steps.
        # Passing t_enc + 1 makes it denoise exactly t_enc steps, starting from
        # sampler.ddim_timesteps[t_enc - 1].
        timesteps=t_enc + 1,
        log_every_t=ddim_steps + 1,
    )
    return samples


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/home/yiqiuliu/DreamDiffusion/outputs/generation_eval")
    parser.add_argument("--checkpoint_path", type=str, default="/home/yiqiuliu/DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-14-47-42/checkpoint_epoch3.pth")
    parser.add_argument("--pretrain_mbm_path", type=str, default="/home/yiqiuliu/DreamDiffusion_old/pretrains/models/eeg_pretrain_scp/checkpoint.pth")
    parser.add_argument("--pretrain_gm_path", type=str, default="/home/yiqiuliu/DreamDiffusion_old/pretrains")
    parser.add_argument("--gallery_eeg_path", type=str, default="/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion.pt")
    parser.add_argument("--source_split", choices=["train", "test", "visual_test200"], default="test",
                        help="Image/latent gallery used as the initialization source. Use train for fair generation; test is the DreamDiffusion validation split oracle.")
    parser.add_argument("--eval_split", choices=["train", "test", "visual_test200"], default="test",
                        help="EEG split used as query. Here test means DreamDiffusion validation, not VisualEEG's 200-image test.")
    parser.add_argument("--source_h5", type=str, default=None)
    parser.add_argument("--eval_h5", type=str, default=None)
    parser.add_argument("--source_image_features", type=str, default=None)
    parser.add_argument("--visual_test_path", type=str, default=None,
                        help="Path to VisualEEG 200-image test_dreamdiffusion.pt.")
    parser.add_argument("--visual_test_image_root", type=str, default=None)
    parser.add_argument("--encoder_checkpoint_path", type=str, default=str(DEFAULT_ENCODER_CHECKPOINT))
    parser.add_argument("--num_items", type=int, default=10)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.7], help=argparse.SUPPRESS)
    parser.add_argument(
        "--candidate_strategy",
        choices=["top1", "self_if_available"],
        default="top1",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--candidate_batch_size", type=int, default=256)
    parser.add_argument("--vae_encode_batch_size", type=int, default=8)
    parser.add_argument("--compute_metrics", type=parse_bool, default=True,
                        help="Compute SSIM, MSE, and CLIP image similarity for source/generated images.")
    parser.add_argument("--compute_official_metrics", type=parse_bool, default=False,
                        help="Also compute official-style reconstruction metrics from eeg_project_sample_code.ipynb.")
    parser.add_argument("--include_swav", type=parse_bool, default=False,
                        help="Include SwAV official metric. This may require torch.hub download, so it is disabled by default.")
    parser.add_argument("--clip_model_path", type=str,
                        default="/home/yiqiuliu/DreamDiffusion_old/pretrains/models/eeg_pretrain_scp/clip_vit_large_patch14")
    parser.add_argument("--max_source_items", type=int, default=None,
                        help="Optional smoke-test limit for the source gallery. Leave unset for real experiments.")
    parser.add_argument("--skip_generation", type=parse_bool, default=False,
                        help="Only write source-candidate CSV; do not load LDM or sample images.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    bank_defaults = split_defaults(args.source_split)
    query_defaults = split_defaults(args.eval_split)
    bank_h5_path = args.source_h5 or bank_defaults.get("h5")
    query_h5_path = args.eval_h5 or query_defaults.get("h5")
    bank_features_path = args.source_image_features or bank_defaults["features"]

    bank_h5 = load_split_data(
        args.source_split,
        h5_override=bank_h5_path if args.source_h5 else None,
        pt_override=args.visual_test_path,
        image_root_override=args.visual_test_image_root,
    )
    query_h5 = load_split_data(
        args.eval_split,
        h5_override=query_h5_path if args.eval_h5 else None,
        pt_override=args.visual_test_path,
        image_root_override=args.visual_test_image_root,
    )
    if args.eval_split == "visual_test200" or args.source_split == "visual_test200":
        gallery_data = bank_h5["dataset"] if args.source_split == "visual_test200" else query_h5["dataset"]
        images = bank_h5["images"] if args.source_split == "visual_test200" else query_h5["images"]
    else:
        gallery_data, images = load_gallery_eeg(args.gallery_eeg_path)
    feature_map = load_multiblur_features(bank_features_path)
    encoder = load_visual_eeg_encoder(args.encoder_checkpoint_path, 63, 1024, 250, device)
    bank = build_candidate_gallery(
        bank_h5,
        gallery_data,
        images,
        feature_map,
        encoder,
        device,
        batch_size=args.candidate_batch_size,
        max_source_items=args.max_source_items,
    )
    print(f"[gallery] {len(bank['indices'])} {args.source_split}-split source candidates")

    query_rows = evenly_spaced_indices(len(query_h5["indices"]), args.num_items)
    query_eeg = torch.stack([
        gallery_data[int(query_h5["indices"][row])]["eeg"].float()
        for row in query_rows
    ], dim=0).to(device)
    query_features = F.normalize(encoder(query_eeg), dim=-1).cpu()
    sim = query_features @ F.normalize(bank["img_features"], dim=-1).T
    top1 = sim.argmax(dim=1)

    model = None
    sampler = None
    clip_metric = None
    if not args.skip_generation:
        model = load_ldm(args, device)
        sampler = DDIMSampler(model)
        if args.compute_metrics:
            clip_metric = ClipImageSimilarity(args.clip_model_path, device)
        if bank["latents"] is None:
            bank["latents"] = encode_image_latents(
                model,
                [bank_h5["image_paths"][row] for row in bank["rows"]],
                device,
                batch_size=args.vae_encode_batch_size,
            )
        if query_h5.get("vae_latents") is None:
            query_h5["vae_latents"] = encode_image_latents(
                model,
                query_h5["image_paths"],
                device,
                batch_size=args.vae_encode_batch_size,
            )

    metric_records = []
    gt_metric_images = []
    variant_metric_images = {}
    run_name = f"{args.eval_split}_generation"
    csv_path = Path(args.output_dir) / f"generation_{run_name}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row", "eval_split", "query_original_idx"],
        )
        writer.writeheader()

        for i, (query_row, top1_bank_pos) in enumerate(zip(query_rows, top1.tolist())):
            query_original_idx = int(query_h5["indices"][query_row])
            target_positions = np.flatnonzero(bank["indices"] == query_original_idx)
            target_bank_pos = int(target_positions[0]) if len(target_positions) else None

            selected_bank_pos = top1_bank_pos
            if args.candidate_strategy == "self_if_available" and target_bank_pos is not None:
                selected_bank_pos = target_bank_pos

            writer.writerow({
                "row": i,
                "eval_split": args.eval_split,
                "query_original_idx": query_original_idx,
            })

            if args.skip_generation:
                continue

            gt_latent = query_h5["vae_latents"][query_row:query_row + 1].to(device)
            init_latent = bank["latents"][selected_bank_pos:selected_bank_pos + 1].to(device)
            cond, _ = model.get_learned_conditioning(query_eeg[i:i + 1])
            noise = torch.randn_like(init_latent)

            gt_img = decode_latent(model, gt_latent).cpu()
            variants = []
            for idx, strength in enumerate(args.strengths):
                sample = ddim_latent_refinement(model, sampler, init_latent, cond, strength, args.ddim_steps, noise)
                variant_name = "generated" if idx == 0 else f"generated_{idx + 1}"
                variants.append((variant_name, decode_latent(model, sample).cpu()))

            if args.compute_metrics or args.compute_official_metrics:
                gt_metric_images.append(gt_img)
                for variant_name, pred_img in variants:
                    variant_metric_images.setdefault(variant_name, []).append(pred_img)

            if args.compute_metrics:
                gt_uint8 = tensor_to_uint8_hwc(gt_img)
                pred_uint8_list = [tensor_to_uint8_hwc(img) for _, img in variants]
                clip_scores = clip_metric.score_against_gt(gt_uint8, pred_uint8_list)
                for (variant_name, pred_img), pred_uint8, clip_score in zip(variants, pred_uint8_list, clip_scores):
                    metric_records.append({
                        "row": i,
                        "query_original_idx": query_original_idx,
                        "variant": variant_name,
                        "ssim": compute_ssim(gt_uint8, pred_uint8),
                        "mse": compute_mse(gt_uint8, pred_uint8),
                        "clip_similarity": clip_score,
                        "eval_pixcorr": compute_eval_pixcorr(gt_img, pred_img),
                        "eval_ssim": compute_eval_ssim(gt_img, pred_img),
                    })

    if metric_records:
        metrics_csv_path = Path(args.output_dir) / f"generation_{run_name}_metrics_by_sample.csv"
        with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_records[0].keys()))
            writer.writeheader()
            writer.writerows(metric_records)

        metrics_summary_path = Path(args.output_dir) / f"generation_{run_name}_metrics_summary.json"
        metrics_summary = summarize_metric_records(metric_records)
        if args.compute_official_metrics and gt_metric_images and variant_metric_images:
            if device.type == "cuda":
                if model is not None:
                    model.to("cpu")
                encoder.to("cpu")
                torch.cuda.empty_cache()
            metrics_summary["official_eval"] = compute_official_identification_metrics(
                torch.cat(gt_metric_images, dim=0),
                {
                    variant_name: torch.cat(images, dim=0)
                    for variant_name, images in variant_metric_images.items()
                },
                args.clip_model_path,
                device,
                include_swav=args.include_swav,
            )
        with metrics_summary_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        print(f"[done] wrote {metrics_csv_path}")
        print(f"[done] wrote {metrics_summary_path}")
    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
