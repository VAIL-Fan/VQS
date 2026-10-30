"""
Fast DRM scanner: iterate all videos, run inference only on segments with GT objects,
skip visualization entirely. Output a text file listing videos with non-empty DRM.

Usage (4 GPUs):
  CUDA_VISIBLE_DEVICES=0 python tools/scan_drm_videos.py --shard 0 --num_shards 4 &
  CUDA_VISIBLE_DEVICES=1 python tools/scan_drm_videos.py --shard 1 --num_shards 4 &
  CUDA_VISIBLE_DEVICES=2 python tools/scan_drm_videos.py --shard 2 --num_shards 4 &
  CUDA_VISIBLE_DEVICES=3 python tools/scan_drm_videos.py --shard 3 --num_shards 4 &
Then merge: cat drm_videos_shard*.txt | sort -u > drm_videos_all.txt
"""

import argparse
import logging
import os
import sys

# Ensure the project root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

from training.dataset.vos_raw_dataset import PNGRawDataset
from training.utils.data_utils import collate_fn, Frame, Object, VideoDatapoint
from training.dataset.transforms import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI

logger = logging.getLogger("scan_drm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


class TestChunkSampler:
    def __init__(self, num_frames=8):
        self.num_frames = num_frames

    def sample_chunks(self, video, segment_loader):
        chunks = []
        num_video_frames = len(video.frames)
        if num_video_frames < 2:
            return chunks, []
        query_frame_idx = 0
        first_frame_segments = segment_loader.load(video.frames[query_frame_idx].frame_idx)
        object_ids = sorted(
            [oid for oid in first_frame_segments.keys() if first_frame_segments[oid].sum() > 0]
        )
        if not object_ids:
            return chunks, object_ids
        query_frame = video.frames[query_frame_idx]
        video_frame_start = 1
        while video_frame_start < num_video_frames:
            video_frame_end = min(video_frame_start + (self.num_frames - 1), num_video_frames)
            video_frame_indices = list(range(video_frame_start, video_frame_end))
            if len(video_frame_indices) < (self.num_frames - 1):
                last = video_frame_indices[-1] if video_frame_indices else video_frame_start
                while len(video_frame_indices) < (self.num_frames - 1):
                    video_frame_indices.append(last)
            chunk_frames = [query_frame] + [video.frames[i] for i in video_frame_indices]
            chunks.append(chunk_frames)
            video_frame_start = video_frame_end
        return chunks, object_ids


def has_gt_object(segment_loader, frame_idx, object_ids):
    """Check if any object has a non-empty GT mask at this frame."""
    segs = segment_loader.load(frame_idx)
    for obj_id in object_ids:
        mask = segs.get(obj_id, None)
        if mask is not None:
            m = mask.numpy() if hasattr(mask, 'numpy') else np.array(mask)
            if m.sum() > 0:
                return True
    return False


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def scan_video(model, video_dataset, video_idx, video_name, object_ids_override=None,
               num_frames_per_segment=8, transforms=None):
    """
    Run inference on one video (no visualization). Returns True if DRM is non-empty.
    Only processes segments where at least one non-query frame has GT objects.
    """
    device = next(model.parameters()).device

    video, segment_loader = video_dataset.get_video(video_idx)
    chunk_sampler = TestChunkSampler(num_frames=num_frames_per_segment)
    chunks, object_ids = chunk_sampler.sample_chunks(video, segment_loader)
    if not chunks:
        return False

    # Preload frames
    frame_cache = {}
    for frame in video.frames:
        img = Image.open(frame.image_path).convert("RGB")
        frame_cache[frame.frame_idx] = img

    query_frame_0_segments = segment_loader.load(0)

    def build_datapoint(chunk_frames):
        images = []
        for local_idx, frame in enumerate(chunk_frames):
            rgb = frame_cache[frame.frame_idx]
            w, h = rgb.size
            images.append(Frame(data=rgb, objects=[]))
            if local_idx == 0:
                segments = query_frame_0_segments
            else:
                segments = segment_loader.load(frame.frame_idx)
            for obj_id in object_ids:
                if obj_id in segments:
                    seg = segments[obj_id].to(torch.uint8)
                else:
                    seg = torch.zeros(h, w, dtype=torch.uint8)
                images[local_idx].objects.append(
                    Object(object_id=obj_id, frame_index=frame.frame_idx, segment=seg)
                )
        return VideoDatapoint(frames=images, video_id=0, size=(h, w))

    all_video_drm = defaultdict(list)

    for seg_idx, chunk_frames in enumerate(chunks):
        frame_indices = [f.frame_idx for f in chunk_frames]

        # Skip segments where NO non-query frame has GT objects
        has_obj = False
        for t in range(1, len(chunk_frames)):
            if has_gt_object(segment_loader, chunk_frames[t].frame_idx, object_ids):
                has_obj = True
                break
        if not has_obj:
            continue

        video_datapoint = build_datapoint(chunk_frames)
        if transforms:
            video_datapoint = transforms(video_datapoint, epoch=0)
        batched = collate_fn([video_datapoint], dict_key="all")
        batched = batched.to(device)

        backbone_out = model.forward_image(batched.flat_img_batch)
        model.prepare_prompt_inputs(backbone_out, batched, start_frame_idx=0)
        backbone_out["init_cond_frames"] = [0]
        backbone_out["frames_to_add_correction_pt"] = []
        backbone_out["use_pt_input"] = False

        if 0 in backbone_out["gt_masks_per_frame"]:
            query_masks = backbone_out["gt_masks_per_frame"][0]
            obj_to_frame_idx = batched.obj_to_frame_idx[0]
            video_indices = obj_to_frame_idx[:, 1].to(torch.long)
            batch_mask_list = []
            for vid_idx in range(batched.img_batch.shape[1]):
                vid_mask = query_masks[video_indices == vid_idx]
                if vid_mask.numel() == 0:
                    zeros = torch.zeros(
                        (1, 1, query_masks.shape[-2], query_masks.shape[-1]),
                        dtype=query_masks.dtype, device=query_masks.device,
                    ).float()
                    batch_mask_list.append(zeros)
                else:
                    batch_mask_list.append(vid_mask.any(dim=0, keepdim=True).float())
            if batch_mask_list:
                backbone_out["mask_inputs_per_frame"] = {0: torch.cat(batch_mask_list, dim=0)}

        # Stage 1 only (no attention hooks, no stage 2)
        stage1_outputs = model.forward_tracking(backbone_out, batched, return_dict=True)

        # Patch IoU with actual GT IoU
        frame_ids = sorted(stage1_outputs["non_cond_frame_outputs"].keys())
        B = len(object_ids)
        for t in frame_ids:
            out = stage1_outputs["non_cond_frame_outputs"][t]
            logits_list = out.get("multistep_pred_multimasks_high_res", None)
            if not logits_list or logits_list[0] is None:
                continue
            logits = logits_list[0]
            Bm, M, H_pred, W_pred = logits.shape
            chunk_frame = chunk_frames[t] if t < len(chunk_frames) else None
            if chunk_frame is None:
                continue
            gt_segs = segment_loader.load(chunk_frame.frame_idx)
            actual_ious = torch.zeros(Bm, M, device=device, dtype=torch.float32)
            for b_idx, obj_id in enumerate(object_ids):
                if b_idx >= Bm:
                    break
                gt_mask = gt_segs.get(obj_id, None)
                if gt_mask is None:
                    continue
                gt_bin = (gt_mask > 0).to(device).float()
                gt_resized = F.interpolate(
                    gt_bin.unsqueeze(0).unsqueeze(0),
                    size=(H_pred, W_pred), mode="nearest"
                )[0, 0] > 0.5
                for m in range(M):
                    pred_bin = logits[b_idx, m] > 0
                    inter = (pred_bin & gt_resized).sum().float()
                    union = (pred_bin | gt_resized).sum().float()
                    actual_ious[b_idx, m] = inter / (union + 1e-6)
            ious_list = out.get("multistep_pred_ious", None)
            if ious_list and ious_list[0] is not None:
                ious_list[0][:] = actual_ious

        # Build override memory (triggers DRM scoring)
        override_memory = model._build_stage_two_override_memory(stage1_outputs, backbone_out, batched)

        if override_memory is not None and "dam_memory_types_per_obj" in override_memory:
            frame_ids_sorted = sorted(stage1_outputs["non_cond_frame_outputs"].keys())
            dam_types = override_memory["dam_memory_types_per_obj"]
            B = len(dam_types)

            if model.dam_drm_size > 0 and frame_ids_sorted:
                strengths = torch.full((B, len(frame_ids_sorted)), -1e9, device=device, dtype=torch.float32)
                has_mat = torch.zeros((B, len(frame_ids_sorted)), device=device, dtype=torch.bool)
                for j, t in enumerate(frame_ids_sorted):
                    out = stage1_outputs["non_cond_frame_outputs"][t]
                    strength, has = model._score_predicted_distractor_top1_low_pred(out)
                    if strength is not None and has is not None:
                        strengths[:, j] = torch.where(has, strength, torch.tensor(-1e9, device=device))
                        has_mat[:, j] = has

                if model.dam_drm_selection == "max_divergence":
                    k = min(model.dam_drm_size, len(frame_ids_sorted))
                    top_vals, top_idx = torch.topk(strengths, k=k, dim=1)
                    for b_idx in range(B):
                        for j in range(k):
                            if top_vals[b_idx, j] > -1e8:
                                actual_frame_idx = frame_indices[int(top_idx[b_idx, j].item())] if int(top_idx[b_idx, j].item()) < len(frame_indices) else int(top_idx[b_idx, j].item())
                                all_video_drm[b_idx].append((seg_idx, actual_frame_idx))
                else:
                    used = torch.zeros_like(has_mat)
                    for _ in range(model.dam_drm_size):
                        for b_idx in range(B):
                            idxs = torch.where(has_mat[b_idx] & ~used[b_idx])[0]
                            if idxs.numel() > 0:
                                chosen = idxs.max()
                                used[b_idx, chosen] = True
                                t_val = int(chosen.item())
                                actual_frame_idx = frame_indices[frame_ids_sorted[t_val]] if frame_ids_sorted[t_val] < len(frame_indices) else frame_ids_sorted[t_val]
                                all_video_drm[b_idx].append((seg_idx, actual_frame_idx))

        # Early exit: if we already found DRM, no need to process more segments
        n_total_drm = sum(len(v) for v in all_video_drm.values())
        if n_total_drm > 0:
            return True

    n_total_drm = sum(len(v) for v in all_video_drm.values())
    return n_total_drm > 0


def main():
    parser = argparse.ArgumentParser(description="Fast DRM video scanner")
    parser.add_argument("--sam2_cfg", type=str,
        default="sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml")
    parser.add_argument("--sam2_checkpoint", type=str,
        default="./sam2_logs/two_stage_mem_dam_adaptive_gating_Jan28/checkpoints/checkpoint.pt")
    parser.add_argument("--base_video_dir", type=str,
        default="./data/VQS-4K/images")
    parser.add_argument("--input_mask_dir", type=str,
        default="./data/VQS-4K/masks")
    parser.add_argument("--num_frames_per_segment", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--output_file", type=str, default=None,
        help="Output file for DRM video list (default: drm_videos_shard{shard}.txt)")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"drm_videos_shard{args.shard:02d}.txt"

    # Load model
    if os.path.isfile(args.sam2_cfg):
        cfg = OmegaConf.load(args.sam2_cfg)
    else:
        GlobalHydra.instance().clear()
        with initialize_config_module("sam2", version_base="1.2"):
            cfg = compose(config_name=args.sam2_cfg)

    model = instantiate(cfg.trainer.model, _recursive_=True)
    ckpt = torch.load(args.sam2_checkpoint, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.pred_obj_scores = False
    model.distractor_pred_min_score = 0.5
    logger.info(f"Model loaded on {device} (distractor_pred_min_score=0.5)")

    # Transforms
    eval_transforms = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=cfg.scratch.resolution, square=True, consistent_transform=True),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset and shard
    video_dataset = PNGRawDataset(
        img_folder=args.base_video_dir,
        gt_folder=args.input_mask_dir,
        file_list_txt=None,
        is_palette=True,
    )
    all_names = list(video_dataset.video_names)
    total = len(all_names)

    # Shard
    shard_size = (total + args.num_shards - 1) // args.num_shards
    start = args.shard * shard_size
    end = min(start + shard_size, total)
    shard_names = all_names[start:end]
    logger.info(f"Shard {args.shard}/{args.num_shards}: videos {start}-{end-1} ({len(shard_names)} videos)")

    # Build name->idx map
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}

    drm_videos = []
    for i, video_name in enumerate(shard_names):
        video_idx = name_to_idx[video_name]
        try:
            has_drm = scan_video(
                model=model,
                video_dataset=video_dataset,
                video_idx=video_idx,
                video_name=video_name,
                num_frames_per_segment=args.num_frames_per_segment,
                transforms=eval_transforms,
            )
            status = "DRM_FOUND" if has_drm else "no_drm"
            if has_drm:
                drm_videos.append(video_name)
                # Write immediately so we can see partial results
                with open(args.output_file, "a") as f:
                    f.write(video_name + "\n")
            if (i + 1) % 10 == 0 or has_drm:
                logger.info(f"[{i+1}/{len(shard_names)}] {video_name}: {status} (total DRM videos: {len(drm_videos)})")
        except Exception as e:
            logger.error(f"[{i+1}/{len(shard_names)}] {video_name}: ERROR {e}")

    logger.info(f"\nDone. {len(drm_videos)} videos with DRM out of {len(shard_names)} scanned.")
    logger.info(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()
