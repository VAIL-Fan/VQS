#!/usr/bin/env python3
"""
Visualization Inference Script for DAM Adaptive Gating Model (Jan28)
====================================================================

This script runs inference on specified videos and produces three types of
visualizations designed for paper figures:

1. **Mask Decoder Cross-Attention Heatmaps** – Inside the stage-2 mask
   decoder, a TwoWayTransformer uses cross-attention so that mask tokens
   attend to the memory-conditioned image features.  The script hooks
   into every ``cross_attn_token_to_image`` (and the final token-to-image
   attention) inside ``sam_mask_decoder.transformer`` and captures the
   attention weights.  The resulting spatial attention map highlights
   which regions of the current frame the decoder focuses on when
   predicting the segmentation mask.

2. **RAM (Recent Appearance Memory) Selection** – For each non-query frame
   processed in stage 2, the script logs which past frames were selected
   into the RAM buffer and overlays the RAM memory mask on the
   corresponding source frames.

3. **DRM (Distractor Reference Memory) Selection** – Similarly, the DRM
   slot selections are visualized on their source frames, showing which
   regions the model identified as distractors.

The script is **standalone**: it does not modify any existing model code.
It hooks into the model's internal tensors via a lightweight wrapper around
``_build_stage_two_override_memory`` and ``_process_video_frames_parallel``.

Usage
-----
  python tools/visualize_dam_inference.py \\
      --sam2_cfg <training_config_yaml> \\
      --sam2_checkpoint <checkpoint_path> \\
      --base_video_dir <images_folder> \\
      --input_mask_dir <masks_folder> \\
      --video_list_file <txt_file_with_video_names> \\
      --output_vis_dir <output_folder>
"""

import argparse
import os
import sys
import math
import logging
from collections import defaultdict

# Ensure the project root is on sys.path so Hydra can resolve sam2.modeling.* targets
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

# -- Project imports --
from training.dataset.vos_raw_dataset import PNGRawDataset
from training.dataset.vos_dataset import VOSDataset, load_images
from training.dataset.vos_segment_loader import PalettisedPNGSegmentLoader
from training.utils.data_utils import collate_fn, Frame, Object, VideoDatapoint
from training.model.sam2 import SAM2Train
from training.dataset.transforms import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Cross-attention map capture
# ──────────────────────────────────────────────────────────────────────────────

class AttentionMapCapture:
    """
    Hooks into the mask decoder's TwoWayTransformer cross-attention layers
    (``cross_attn_token_to_image``) to capture attention weights during
    ``sam_mask_decoder.forward(...)``.

    In each ``TwoWayAttentionBlock`` the mask / IoU / object-score tokens
    (queries) attend to the memory-conditioned image features (keys, shape
    H*W).  There is also a ``final_attn_token_to_image`` in the outer
    ``TwoWayTransformer``.  We hook into all of them.

    After a forward pass the captured maps are in ``self.attn_maps``, one
    per hook call.  Each entry has shape ``[B, num_heads, N_tokens, H*W]``.

    Usage::

        cap = AttentionMapCapture(model)
        cap.install()          # register hooks
        cap.clear()            # call before each forward_tracking
        model.forward_tracking(...)   # normal forward
        cap.remove()           # remove hooks when done
    """

    def __init__(self, model):
        self.model = model
        self._hooks = []
        self.attn_maps = []  # list of [B, heads, N_tokens, H*W] per hook call

    def install(self):
        """Register forward hooks on mask-decoder cross-attention modules."""
        transformer = self.model.sam_mask_decoder.transformer
        # Hook each TwoWayAttentionBlock's cross_attn_token_to_image
        for layer in transformer.layers:
            h = layer.cross_attn_token_to_image.register_forward_hook(
                self._make_hook(), with_kwargs=True
            )
            self._hooks.append(h)
        # Hook the final token-to-image attention
        h = transformer.final_attn_token_to_image.register_forward_hook(
            self._make_hook(), with_kwargs=True
        )
        self._hooks.append(h)

    def _make_hook(self):
        capture = self  # closure

        def _hook(module, args, kwargs, output):
            # Attention.forward is called as cross_attn(q=q, k=k, v=keys),
            # so inputs arrive as kwargs, not positional args.
            q_in = kwargs.get('q') if kwargs and 'q' in kwargs else (args[0] if len(args) > 0 else None)
            k_in = kwargs.get('k') if kwargs and 'k' in kwargs else (args[1] if len(args) > 1 else None)
            if q_in is None or k_in is None:
                return

            with torch.no_grad():
                q = module.q_proj(q_in)
                k = module.k_proj(k_in)
                q = module._separate_heads(q, module.num_heads)
                k = module._separate_heads(k, module.num_heads)
                # q: [B, heads, N_tokens, d]   k: [B, heads, H*W, d]
                d = q.shape[-1]
                attn = torch.matmul(
                    q.float(), k.float().transpose(-2, -1)
                ) / math.sqrt(d)
                attn = torch.softmax(attn, dim=-1)  # [B, heads, N_tokens, H*W]
                capture.attn_maps.append(attn.cpu())

        return _hook

    def clear(self):
        self.attn_maps = []

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "jet",
) -> np.ndarray:
    """
    Overlay a [H_heat, W_heat] heatmap on a [H, W, 3] uint8 image.
    Returns [H, W, 3] uint8 image with heatmap overlay.
    """
    H, W = image.shape[:2]
    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    colormap = cm.get_cmap(cmap_name)
    heatmap_color = colormap(heatmap_resized)[:, :, :3]  # [H, W, 3] float in [0,1]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    blended = (alpha * heatmap_color.astype(np.float32)
               + (1 - alpha) * image.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def overlay_mask_contour_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    fill_alpha: float = 0.25,
) -> np.ndarray:
    """Overlay a binary mask as a colored contour + semi-transparent fill on the image."""
    overlay = image.copy()
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Semi-transparent fill
    colored_fill = np.zeros_like(image)
    colored_fill[:] = color
    fill_region = mask_uint8 > 0
    overlay[fill_region] = (
        fill_alpha * colored_fill[fill_region].astype(np.float32)
        + (1 - fill_alpha) * overlay[fill_region].astype(np.float32)
    ).astype(np.uint8)

    # Contour
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def save_image(arr: np.ndarray, path: str):
    """Save a numpy array [H, W, 3] uint8 as an image file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


# ──────────────────────────────────────────────────────────────────────────────
# Chunk Sampler (same as test_video_segmentation.py)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Main visualization
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def visualize_video(
    model,
    base_video_dir,
    input_mask_dir,
    output_vis_dir,
    video_name,
    attn_capture=None,
    num_frames_per_segment=8,
    transforms=None,
    score_thresh=0.5,
    max_segments=0,
    drm_only=False,
):
    """Run inference on one video and generate all visualizations."""
    device = next(model.parameters()).device

    # Dataset loading
    video_dataset = PNGRawDataset(
        img_folder=base_video_dir,
        gt_folder=input_mask_dir,
        file_list_txt=None,
        is_palette=True,
    )
    video_idx = None
    for idx, name in enumerate(video_dataset.video_names):
        if name == video_name:
            video_idx = idx
            break
    if video_idx is None:
        logger.warning(f"Video {video_name} not found, skipping")
        return

    video, segment_loader = video_dataset.get_video(video_idx)
    chunk_sampler = TestChunkSampler(num_frames=num_frames_per_segment)
    chunks, object_ids = chunk_sampler.sample_chunks(video, segment_loader)
    if not chunks:
        logger.warning(f"No valid chunks for {video_name}")
        return

    logger.info(f"  {len(chunks)} segments, {len(object_ids)} objects, {len(video.frames)} frames")

    if max_segments > 0:
        chunks = chunks[:max_segments]
        logger.info(f"  (limited to {max_segments} segments for testing)")

    # Preload frames
    frame_cache = {}
    frame_original_sizes = {}
    for frame in video.frames:
        img = Image.open(frame.image_path).convert("RGB")
        frame_cache[frame.frame_idx] = img
        frame_original_sizes[frame.frame_idx] = img.size

    orig_width, orig_height = frame_original_sizes[video.frames[0].frame_idx]
    query_frame_0_segments = segment_loader.load(0)

    # Output dirs
    heatmap_s1_dir = os.path.join(output_vis_dir, "heatmaps_stage1", video_name)
    heatmap_s2_dir = os.path.join(output_vis_dir, "heatmaps_stage2", video_name)
    heatmap_cmp_dir = os.path.join(output_vis_dir, "heatmaps_comparison", video_name)
    ram_dir = os.path.join(output_vis_dir, "ram_selection", video_name)
    drm_dir = os.path.join(output_vis_dir, "drm_selection", video_name)
    mask_dir = os.path.join(output_vis_dir, "pred_masks", video_name)
    gt_mask_dir = os.path.join(output_vis_dir, "gt_masks", video_name)
    os.makedirs(heatmap_s1_dir, exist_ok=True)
    os.makedirs(heatmap_s2_dir, exist_ok=True)
    os.makedirs(heatmap_cmp_dir, exist_ok=True)
    os.makedirs(ram_dir, exist_ok=True)
    os.makedirs(drm_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(gt_mask_dir, exist_ok=True)

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

    # ── Video-level accumulators (across ALL segments) ──
    # ram_info_all[obj_b] = [(seg_idx, actual_frame_idx, mask_np), ...]
    # drm_info_all[obj_b] = [(seg_idx, actual_frame_idx, mask_np), ...]
    all_video_ram = defaultdict(list)
    all_video_drm = defaultdict(list)

    for seg_idx, chunk_frames in enumerate(chunks):
        frame_indices = [f.frame_idx for f in chunk_frames]
        logger.info(f"    Segment {seg_idx+1}/{len(chunks)}: query={frame_indices[0]}, frames={frame_indices[1:]}")

        video_datapoint = build_datapoint(chunk_frames)
        if transforms:
            video_datapoint = transforms(video_datapoint, epoch=0)

        batched = collate_fn([video_datapoint], dict_key="all")
        batched = batched.to(device)

        # Forward image encoder
        backbone_out = model.forward_image(batched.flat_img_batch)
        model.prepare_prompt_inputs(backbone_out, batched, start_frame_idx=0)
        backbone_out["init_cond_frames"] = [0]
        backbone_out["frames_to_add_correction_pt"] = []
        backbone_out["use_pt_input"] = False

        # Set up query mask
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

        # ── Stage 1 ──
        attn_capture.clear()  # Clear before stage-1 so captured maps are fresh
        stage1_outputs = model.forward_tracking(backbone_out, batched, return_dict=True)
        stage1_captured = list(attn_capture.attn_maps)  # save stage-1 decoder attn maps

        # ── Patch multistep_pred_ious with actual IoU (pred vs GT) ──
        # The IoU prediction head is collapsed (outputs ~0 for all checkpoints).
        # Replace pred_ious with actual mask IoU to enable meaningful RAM/DRM selection.
        frame_ids = sorted(stage1_outputs["non_cond_frame_outputs"].keys())
        B = len(object_ids)
        for t in frame_ids:
            out = stage1_outputs["non_cond_frame_outputs"][t]
            logits_list = out.get("multistep_pred_multimasks_high_res", None)
            if not logits_list or logits_list[0] is None:
                continue
            logits = logits_list[0]  # [B, M, H, W]
            Bm, M, H_pred, W_pred = logits.shape
            # Load GT mask for this frame
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
                )[0, 0] > 0.5  # [H_pred, W_pred] bool
                for m in range(M):
                    pred_bin = logits[b_idx, m] > 0  # [H_pred, W_pred]
                    inter = (pred_bin & gt_resized).sum().float()
                    union = (pred_bin | gt_resized).sum().float()
                    actual_ious[b_idx, m] = inter / (union + 1e-6)
            # Patch the pred_ious in-place
            out["multistep_pred_ious"][0] = actual_ious

        # ── Extract RAM/DRM selections from stage-2 override memory ──
        seg_ram_info = {}  # {obj_b: [(actual_frame_idx, mask_np), ...]}
        seg_drm_info = {}  # {obj_b: [(actual_frame_idx, mask_np), ...]}
        override_memory = model._build_stage_two_override_memory(stage1_outputs, backbone_out, batched)

        if override_memory is not None and "dam_memory_types_per_obj" in override_memory:
            dam_types = override_memory["dam_memory_types_per_obj"]
            B = len(dam_types)
            frame_ids_sorted = sorted(stage1_outputs["non_cond_frame_outputs"].keys())

            for b in range(B):
                seg_ram_info[b] = []
                seg_drm_info[b] = []

            # ── RAM: threshold-based selection using actual IoU ──
            if model.dam_ram_size > 0 and frame_ids_sorted:
                scores = []
                for t in frame_ids_sorted:
                    s = model._get_stage1_frame_scores(stage1_outputs["non_cond_frame_outputs"][t])
                    if s is None:
                        s = torch.zeros(B, device=device, dtype=torch.float32)
                    scores.append(s)
                scores_tensor = torch.stack(scores, dim=1)  # [B, F]
                eligible = scores_tensor >= model.dam_ram_min_score

                ram_frames_per_obj = [[] for _ in range(B)]

                if not model.enable_quality_weighted_ram:
                    # Greedy recency: pick most-recent eligible frames
                    used = torch.zeros_like(eligible)
                    for _ in range(model.dam_ram_size):
                        for b_idx in range(B):
                            idxs = torch.where(eligible[b_idx] & ~used[b_idx])[0]
                            if idxs.numel() > 0:
                                chosen = idxs.max()  # most recent
                                used[b_idx, chosen] = True
                                ram_frames_per_obj[b_idx].append(frame_ids_sorted[int(chosen.item())])
                else:
                    # Quality-weighted selection with threshold
                    quality_normalized = torch.zeros_like(scores_tensor)
                    for b_idx in range(B):
                        eligible_scores = scores_tensor[b_idx][eligible[b_idx]]
                        if eligible_scores.numel() == 0:
                            continue
                        q_min, q_max = eligible_scores.min(), eligible_scores.max()
                        if q_max > q_min:
                            quality_normalized[b_idx] = (scores_tensor[b_idx] - q_min) / (q_max - q_min + 1e-8)
                        else:
                            quality_normalized[b_idx] = eligible[b_idx].float()
                    frame_positions = torch.tensor(frame_ids_sorted, device=device, dtype=torch.float32)
                    T_max = frame_positions.max()
                    recency = torch.exp(-model.ram_recency_decay * (T_max - frame_positions))
                    rec_min, rec_max = recency.min(), recency.max()
                    recency_norm = (recency - rec_min) / (rec_max - rec_min + 1e-8) if rec_max > rec_min else recency
                    alpha = model.ram_selection_alpha
                    sel_scores = alpha * recency_norm.unsqueeze(0).expand(B, -1) + (1 - alpha) * quality_normalized
                    sel_scores = sel_scores.masked_fill(~eligible, -1e9)
                    k = min(model.dam_ram_size, len(frame_ids_sorted))
                    if k > 0:
                        topk_vals, topk_idx = torch.topk(sel_scores, k=k, dim=1)
                        for b_idx in range(B):
                            for j in range(k):
                                if topk_vals[b_idx, j] > -1e8:  # passed threshold
                                    ram_frames_per_obj[b_idx].append(frame_ids_sorted[int(topk_idx[b_idx, j].item())])

                n_ram_selected = sum(len(v) for v in ram_frames_per_obj)
                # Log all actual IoU scores for transparency
                score_strs = [f"t{frame_ids_sorted[i]}={float(scores_tensor[0,i]):.4f}" for i in range(len(frame_ids_sorted))]
                logger.info(f"      RAM actual-IoU: [{', '.join(score_strs)}] (threshold={model.dam_ram_min_score})")
                if n_ram_selected > 0:
                    selected_frames = [f for obj_frames in ram_frames_per_obj for f in obj_frames]
                    logger.info(f"      RAM: {n_ram_selected} frame(s) selected → local_t={selected_frames}")
                else:
                    logger.info(f"      RAM: 0 frames (all actual-IoU below threshold)")

                for b_idx in range(B):
                    for t in ram_frames_per_obj[b_idx]:
                        out = stage1_outputs["non_cond_frame_outputs"][t]
                        masks_hr = out.get("pred_masks_high_res", None)
                        if masks_hr is not None:
                            mask_np = (torch.sigmoid(masks_hr[b_idx, 0]).cpu().numpy() > 0.5).astype(np.uint8) * 255
                        else:
                            mask_np = None
                        actual_frame_idx = frame_indices[t] if t < len(frame_indices) else t
                        seg_ram_info[b_idx].append((actual_frame_idx, mask_np))

            # ── DRM: re-derive frame selections using actual IoU ──
            # The patched multistep_pred_ious now contain actual IoU values,
            # so _score_predicted_distractor_top1_low_pred uses meaningful scores.
            if model.dam_drm_size > 0 and frame_ids_sorted:
                drm_frames_per_obj = [[] for _ in range(B)]
                strengths = torch.full((B, len(frame_ids_sorted)), -1e9, device=device, dtype=torch.float32)
                has_mat = torch.zeros((B, len(frame_ids_sorted)), device=device, dtype=torch.bool)
                for j, t in enumerate(frame_ids_sorted):
                    out = stage1_outputs["non_cond_frame_outputs"][t]
                    strength, has = model._score_predicted_distractor_top1_low_pred(out)
                    if strength is not None and has is not None:
                        # Only store strength for frames that actually HAVE a distractor
                        strengths[:, j] = torch.where(has, strength, torch.tensor(-1e9, device=device))
                        has_mat[:, j] = has

                if model.dam_drm_selection == "max_divergence":
                    k = min(model.dam_drm_size, len(frame_ids_sorted))
                    top_vals, top_idx = torch.topk(strengths, k=k, dim=1)
                    for b_idx in range(B):
                        for j in range(k):
                            if top_vals[b_idx, j] > -1e8:
                                drm_frames_per_obj[b_idx].append(frame_ids_sorted[int(top_idx[b_idx, j].item())])
                else:
                    used = torch.zeros_like(has_mat)
                    for _ in range(model.dam_drm_size):
                        for b_idx in range(B):
                            idxs = torch.where(has_mat[b_idx] & ~used[b_idx])[0]
                            if idxs.numel() > 0:
                                chosen = idxs.max()
                                used[b_idx, chosen] = True
                                drm_frames_per_obj[b_idx].append(frame_ids_sorted[int(chosen.item())])

                n_drm_selected = sum(len(v) for v in drm_frames_per_obj)
                # Log DRM strengths (now based on actual IoU)
                str_strs = [f"t{frame_ids_sorted[i]}={float(strengths[0,i]):.4f}" for i in range(len(frame_ids_sorted))]
                logger.info(f"      DRM strengths (actual-IoU): [{', '.join(str_strs)}]")
                selected_drm = [f for obj_frames in drm_frames_per_obj for f in obj_frames]
                logger.info(f"      DRM: {n_drm_selected} frame(s) selected → local_t={selected_drm}")

                for b_idx in range(B):
                    for t in drm_frames_per_obj[b_idx]:
                        out = stage1_outputs["non_cond_frame_outputs"][t]
                        dmask, has = model._select_predicted_distractor_mask_top1_low_pred(out)
                        if dmask is not None and has is not None and bool(has[b_idx].item()):
                            mask_np = (dmask[b_idx, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                        else:
                            mask_np = None
                        actual_frame_idx = frame_indices[t] if t < len(frame_indices) else t
                        seg_drm_info[b_idx].append((actual_frame_idx, mask_np))

        # ── Accumulate to video-level ──
        for b_idx in seg_ram_info:
            for actual_frame_idx, mask_np in seg_ram_info[b_idx]:
                all_video_ram[b_idx].append((seg_idx, actual_frame_idx, mask_np))
        for b_idx in seg_drm_info:
            for actual_frame_idx, mask_np in seg_drm_info[b_idx]:
                all_video_drm[b_idx].append((seg_idx, actual_frame_idx, mask_np))

        # ── Stage 2 (with feature capture) ──
        outputs = model._run_second_stage(backbone_out, batched, stage1_outputs)

        if override_memory is not None:
            if model.enable_distractor_memory and model.distractor_use_in_stage2:
                q0 = stage1_outputs["cond_frame_outputs"].get(0, None)
                if q0 is not None and "distractor_maskmem_features" in q0 and "distractor_maskmem_pos_enc" in q0:
                    override_memory["distractor_maskmem_features"] = q0["distractor_maskmem_features"]
                    override_memory["distractor_maskmem_pos_enc"] = q0["distractor_maskmem_pos_enc"]
            cached = stage1_outputs.get("backbone_cache")
            # Clear and re-capture attention maps during stage-2 forward
            attn_capture.clear()
            stage2_dict = model.forward_tracking(
                backbone_out, batched,
                return_dict=True,
                cached_backbone=cached,
                override_cond_outputs=override_memory,
            )
        else:
            # No override memory -> stage-1 attn maps are already captured
            # (do NOT clear, the stage-1 maps are what we want)
            stage2_dict = stage1_outputs

        # ── Per-segment visualizations ──
        num_frames_in_seg = len(frame_indices)

        # Helper: extract a spatial heatmap from a list of captured attention maps
        num_hooks_per_frame = len(model.sam_mask_decoder.transformer.layers) + 1  # 3
        _s = 1 if model.sam_mask_decoder.pred_obj_scores else 0
        _mask_start = _s + 1  # skip obj_score (if any) + iou_token
        _mask_end = _mask_start + model.sam_mask_decoder.num_mask_tokens  # 4 mask tokens

        def _extract_heatmap(captured_maps, local_t):
            """Extract a normalised [H,W] heatmap for a non-cond frame.
            Returns np.ndarray or None."""
            fi = (local_t - 1) * num_hooks_per_frame
            if fi + num_hooks_per_frame > len(captured_maps):
                return None
            frame_maps = captured_maps[fi : fi + num_hooks_per_frame]
            mask_attns = []
            for am in frame_maps:
                tok_end = min(_mask_end, am.shape[2])
                mask_attn = am[:, :, _mask_start:tok_end, :]  # [B, heads, n_mask, H*W]
                mask_attns.append(mask_attn.mean(dim=2))  # avg over mask tokens → [B, heads, H*W]
            stacked = torch.stack(mask_attns, dim=0)  # [L, B, heads, H*W]
            avg = stacked.mean(dim=0).mean(dim=1)  # [B, H*W]
            spatial = avg[0]  # [H*W]
            feat_h = feat_w = int(math.sqrt(spatial.shape[0]))
            heatmap_2d = spatial[:feat_h * feat_w].reshape(feat_h, feat_w).numpy()
            hmin, hmax = heatmap_2d.min(), heatmap_2d.max()
            if hmax - hmin > 1e-8:
                return ((heatmap_2d - hmin) / (hmax - hmin)).astype(np.float32)
            return np.zeros((feat_h, feat_w), dtype=np.float32)

        # 1) Stage-1 & Stage-2 mask-decoder cross-attention heatmaps
        stage2_captured = list(attn_capture.attn_maps)  # stage-2 maps (or stage-1 if no override)

        for local_t in range(1, num_frames_in_seg):
            frame_idx = frame_indices[local_t]
            orig_img = np.array(frame_cache[frame_idx])

            hm_s1 = _extract_heatmap(stage1_captured, local_t)
            hm_s2 = _extract_heatmap(stage2_captured, local_t)

            # ---- Stage-1 heatmap ----
            if hm_s1 is not None:
                ov_s1 = overlay_heatmap_on_image(orig_img, hm_s1, alpha=0.5)
                save_image(ov_s1, os.path.join(heatmap_s1_dir, f"seg{seg_idx:02d}_frame{frame_idx:05d}.png"))
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                axes[0].imshow(orig_img); axes[0].set_title(f"Original (frame {frame_idx})", fontsize=12); axes[0].axis("off")
                axes[1].imshow(hm_s1, cmap="jet"); axes[1].set_title("Stage-1 Attention\n(query only)", fontsize=12); axes[1].axis("off")
                axes[2].imshow(ov_s1); axes[2].set_title("Overlay", fontsize=12); axes[2].axis("off")
                plt.tight_layout()
                fig.savefig(os.path.join(heatmap_s1_dir, f"seg{seg_idx:02d}_frame{frame_idx:05d}_panel.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

            # ---- Stage-2 heatmap ----
            if hm_s2 is not None:
                ov_s2 = overlay_heatmap_on_image(orig_img, hm_s2, alpha=0.5)
                save_image(ov_s2, os.path.join(heatmap_s2_dir, f"seg{seg_idx:02d}_frame{frame_idx:05d}.png"))
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                axes[0].imshow(orig_img); axes[0].set_title(f"Original (frame {frame_idx})", fontsize=12); axes[0].axis("off")
                axes[1].imshow(hm_s2, cmap="jet"); axes[1].set_title("Stage-2 Attention\n(query + memory)", fontsize=12); axes[1].axis("off")
                axes[2].imshow(ov_s2); axes[2].set_title("Overlay", fontsize=12); axes[2].axis("off")
                plt.tight_layout()
                fig.savefig(os.path.join(heatmap_s2_dir, f"seg{seg_idx:02d}_frame{frame_idx:05d}_panel.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

            # ---- Side-by-side comparison panel (stage-1 vs stage-2) ----
            if hm_s1 is not None and hm_s2 is not None:
                fig, axes = plt.subplots(1, 4, figsize=(24, 5))
                axes[0].imshow(orig_img); axes[0].set_title(f"Original (frame {frame_idx})", fontsize=12); axes[0].axis("off")
                axes[1].imshow(hm_s1, cmap="jet", vmin=0, vmax=1); axes[1].set_title("Stage-1 Attention\n(query only)", fontsize=12); axes[1].axis("off")
                axes[2].imshow(hm_s2, cmap="jet", vmin=0, vmax=1); axes[2].set_title("Stage-2 Attention\n(query + memory)", fontsize=12); axes[2].axis("off")
                # Difference map: stage2 - stage1
                diff = hm_s2.astype(np.float64) - hm_s1.astype(np.float64)
                axes[3].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1); axes[3].set_title("Difference\n(stage2 − stage1)", fontsize=12); axes[3].axis("off")
                plt.tight_layout()
                fig.savefig(os.path.join(heatmap_cmp_dir, f"seg{seg_idx:02d}_frame{frame_idx:05d}_cmp.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

        # 2) Save predicted masks (from stage 2 outputs)
        for frame_pos, output in enumerate(outputs):
            if frame_pos >= len(frame_indices):
                continue
            frame_idx = frame_indices[frame_pos]
            pred_hr = output.get("pred_masks_high_res", output.get("pred_masks", None))
            if pred_hr is None:
                continue
            if pred_hr.dim() == 3:
                pred_hr = pred_hr.unsqueeze(0)
            for b_idx in range(min(pred_hr.shape[0], 1)):
                mask_probs = torch.sigmoid(pred_hr[b_idx, 0]).cpu().numpy()
                mask_binary = (mask_probs > score_thresh).astype(np.uint8) * 255
                orig_size = frame_original_sizes.get(frame_idx, (orig_width, orig_height))
                mask_img = Image.fromarray(mask_binary).resize(orig_size, Image.NEAREST)
                mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.png")
                mask_img.save(mask_path)

    # ── Save GT mask on clean frames ──
    for frame in video.frames:
        fidx = frame.frame_idx
        orig_img = np.array(frame_cache[fidx])
        gt_segs = segment_loader.load(fidx)
        # Combine all object GT masks into one overlay
        overlay_gt = orig_img.copy()
        has_any = False
        for obj_id in object_ids:
            gt_mask = gt_segs.get(obj_id, None)
            if gt_mask is None:
                continue
            gt_np = gt_mask.numpy() if hasattr(gt_mask, 'numpy') else np.array(gt_mask)
            gt_bin = (gt_np > 0).astype(np.uint8) * 255
            gt_resized = np.array(
                Image.fromarray(gt_bin).resize(
                    (orig_img.shape[1], orig_img.shape[0]), Image.NEAREST
                )
            )
            overlay_gt = overlay_mask_contour_on_image(
                overlay_gt, gt_resized, color=(255, 255, 0), thickness=8, fill_alpha=0.15
            )
            has_any = True
        if has_any:
            save_image(overlay_gt, os.path.join(gt_mask_dir, f"{fidx:05d}.png"))

    # ── After all segments: video-level RAM visualization ──
    n_total_ram = sum(len(v) for v in all_video_ram.values())
    n_total_drm = sum(len(v) for v in all_video_drm.values())
    logger.info(f"  Video total: {n_total_ram} RAM selections, {n_total_drm} DRM selections")

    # If drm_only mode and DRM is empty, clean up empty dirs and skip visualization
    if drm_only and n_total_drm == 0:
        import shutil
        for d in [heatmap_s1_dir, heatmap_s2_dir, heatmap_cmp_dir, ram_dir, drm_dir, mask_dir, gt_mask_dir]:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        logger.info(f"  DRM empty → skipping visualization for {video_name}")
        return

    for b_idx in sorted(all_video_ram.keys()):
        for seg_idx, actual_frame_idx, mask_np in all_video_ram[b_idx]:
            orig_img = np.array(frame_cache[actual_frame_idx])
            if mask_np is not None:
                mask_resized = np.array(
                    Image.fromarray(mask_np).resize(
                        (orig_img.shape[1], orig_img.shape[0]), Image.NEAREST
                    )
                )
                overlay = overlay_mask_contour_on_image(
                    orig_img, mask_resized, color=(0, 200, 0), thickness=8, fill_alpha=0.3
                )
            else:
                overlay = orig_img.copy()
            save_path = os.path.join(
                ram_dir,
                f"seg{seg_idx:02d}_ram_obj{b_idx}_frame{actual_frame_idx:05d}.png",
            )
            save_image(overlay, save_path)

    # ── After all segments: video-level DRM visualization ──
    for b_idx in sorted(all_video_drm.keys()):
        for seg_idx, actual_frame_idx, mask_np in all_video_drm[b_idx]:
            orig_img = np.array(frame_cache[actual_frame_idx])
            if mask_np is not None:
                mask_resized = np.array(
                    Image.fromarray(mask_np).resize(
                        (orig_img.shape[1], orig_img.shape[0]), Image.NEAREST
                    )
                )
                overlay = overlay_mask_contour_on_image(
                    orig_img, mask_resized, color=(255, 50, 50), thickness=8, fill_alpha=0.3
                )
            else:
                overlay = orig_img.copy()
            save_path = os.path.join(
                drm_dir,
                f"seg{seg_idx:02d}_drm_obj{b_idx}_frame{actual_frame_idx:05d}.png",
            )
            save_image(overlay, save_path)

    # ── Video-level summary figure ──
    _create_video_summary(
        video_name, frame_cache, all_video_ram, all_video_drm,
        os.path.join(output_vis_dir, "summary", video_name),
    )


def _create_video_summary(video_name, frame_cache, all_video_ram, all_video_drm, summary_dir):
    """
    Create a composite summary figure for the entire video showing:
    - Query frame (frame 0)
    - All unique RAM selections across all segments (green)
    - All unique DRM selections across all segments (red)
    """
    os.makedirs(summary_dir, exist_ok=True)

    # Deduplicate frames (same frame may appear in multiple segments)
    unique_ram = {}  # frame_idx -> (seg_idx, mask_np)
    unique_drm = {}  # frame_idx -> (seg_idx, mask_np)
    for b_idx in all_video_ram:
        for seg_idx, actual_frame_idx, mask_np in all_video_ram[b_idx]:
            unique_ram[actual_frame_idx] = (seg_idx, mask_np)
    for b_idx in all_video_drm:
        for seg_idx, actual_frame_idx, mask_np in all_video_drm[b_idx]:
            unique_drm[actual_frame_idx] = (seg_idx, mask_np)

    n_ram = len(unique_ram)
    n_drm = len(unique_drm)

    if n_ram == 0 and n_drm == 0:
        # Still create a minimal summary with just the query frame
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if 0 in frame_cache:
            ax.imshow(np.array(frame_cache[0]))
        ax.set_title("Query (Frame 0)\nNo RAM/DRM selected", fontsize=11, fontweight='bold')
        ax.axis("off")
        plt.suptitle(f"{video_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(summary_dir, "memory_selection_summary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # Layout: Query | RAM frames... | DRM frames...
    n_cols = 1 + n_ram + n_drm
    max_cols = 10  # limit width for readability
    if n_cols > max_cols:
        # Use multiple rows
        n_rows = math.ceil(n_cols / max_cols)
        n_cols_per_row = max_cols
    else:
        n_rows = 1
        n_cols_per_row = n_cols

    fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(4.5 * n_cols_per_row, 4.5 * n_rows))
    if n_rows == 1 and n_cols_per_row == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols_per_row == 1:
        axes = axes[:, np.newaxis]

    col = 0
    row = 0

    def next_ax():
        nonlocal row, col
        ax = axes[row, col]
        col += 1
        if col >= n_cols_per_row:
            col = 0
            row += 1
        return ax

    # Query frame
    ax = next_ax()
    if 0 in frame_cache:
        ax.imshow(np.array(frame_cache[0]))
    ax.set_title("Query (Frame 0)", fontsize=10, fontweight='bold')
    ax.axis("off")

    # RAM frames (sorted by frame index)
    for f_idx in sorted(unique_ram.keys()):
        if row >= n_rows:
            break
        ax = next_ax()
        seg_idx, mask_np = unique_ram[f_idx]
        img = np.array(frame_cache.get(f_idx, np.zeros((100, 100, 3), dtype=np.uint8)))
        if mask_np is not None:
            mask_r = np.array(Image.fromarray(mask_np).resize((img.shape[1], img.shape[0]), Image.NEAREST))
            img = overlay_mask_contour_on_image(img, mask_r, color=(0, 200, 0), thickness=8, fill_alpha=0.25)
        ax.imshow(img)
        ax.set_title(f"RAM (Frame {f_idx}, Seg {seg_idx})", fontsize=9, color='green')
        ax.axis("off")

    # DRM frames (sorted by frame index)
    for f_idx in sorted(unique_drm.keys()):
        if row >= n_rows:
            break
        ax = next_ax()
        seg_idx, mask_np = unique_drm[f_idx]
        img = np.array(frame_cache.get(f_idx, np.zeros((100, 100, 3), dtype=np.uint8)))
        if mask_np is not None:
            mask_r = np.array(Image.fromarray(mask_np).resize((img.shape[1], img.shape[0]), Image.NEAREST))
            img = overlay_mask_contour_on_image(img, mask_r, color=(255, 50, 50), thickness=8, fill_alpha=0.25)
        ax.imshow(img)
        ax.set_title(f"DRM (Frame {f_idx}, Seg {seg_idx})", fontsize=9, color='red')
        ax.axis("off")

    # Hide unused axes
    while row < n_rows:
        if col < n_cols_per_row:
            axes[row, col].axis("off")
        col += 1
        if col >= n_cols_per_row:
            col = 0
            row += 1

    plt.suptitle(f"Memory Selection: {video_name}  ({n_ram} RAM, {n_drm} DRM)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(summary_dir, "memory_selection_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Single GPU processing (simple sequential mode)
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Visualize DAM Adaptive Gating inference")
    parser.add_argument(
        "--sam2_cfg", type=str,
        default="sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_RAM1_Feb8.yaml",
        help="Training config YAML",
    )
    parser.add_argument(
        "--sam2_checkpoint", type=str,
        default="./sam2_logs/two_stage_mem_dam_adaptive_gating_ram1_Feb8/checkpoints/checkpoint.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--base_video_dir", type=str,
        default="./data/VQS-4K/images",
        help="Images folder",
    )
    parser.add_argument(
        "--input_mask_dir", type=str,
        default="./data/VQS-4K/masks",
        help="Masks folder",
    )
    parser.add_argument(
        "--video_list_file", type=str,
        default="visualization_video_list.txt",
        help="Text file listing videos to process",
    )
    parser.add_argument(
        "--output_vis_dir", type=str,
        default="./visualization_outputs_RAM1_Feb8",
        help="Output directory for all visualizations",
    )
    parser.add_argument(
        "--num_frames_per_segment", type=int, default=8,
        help="Total frames per segment (matches training config scratch.num_frames)",
    )
    parser.add_argument(
        "--score_thresh", type=float, default=0.5,
        help="Threshold for mask binarization",
    )
    parser.add_argument(
        "--max_segments", type=int, default=0,
        help="Max segments per video (0=all). Useful for quick tests.",
    )
    parser.add_argument(
        "--video_list", nargs="+", default=None,
        help="Inline video names (overrides --video_list_file)",
    )
    parser.add_argument(
        "--drm_only", action="store_true",
        help="Only generate full visualization for videos with non-empty DRM",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ── Load Config ──
    if os.path.isfile(args.sam2_cfg):
        cfg = OmegaConf.load(args.sam2_cfg)
    else:
        GlobalHydra.instance().clear()
        with initialize_config_module("sam2", version_base="1.2"):
            cfg = compose(config_name=args.sam2_cfg)
    
    logger.info(f"Config loaded: {args.sam2_cfg}")
    logger.info(f"Checkpoint: {args.sam2_checkpoint}")

    # ── Load Model ──
    model = instantiate(cfg.trainer.model, _recursive_=True)
    logger.info(f"Loading checkpoint from {args.sam2_checkpoint}")
    ckpt = torch.load(args.sam2_checkpoint, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()
    model.pred_obj_scores = False
    model.distractor_pred_min_score = 0.5
    logger.info(f"Model loaded on {device} (pred_obj_scores bypassed, distractor_pred_min_score→0.5)")

    # ── Install cross-attention hooks ──
    attn_capture = AttentionMapCapture(model)
    attn_capture.install()
    n_hooks = len(model.sam_mask_decoder.transformer.layers) + 1
    logger.info(f"Mask-decoder cross-attention hooks installed ({n_hooks} modules)")

    # ── Transforms ──
    eval_transforms = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=cfg.scratch.resolution, square=True, consistent_transform=True),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Video List ──
    if args.video_list:
        video_names = args.video_list
    elif args.video_list_file and os.path.isfile(args.video_list_file):
        with open(args.video_list_file, "r") as f:
            video_names = [line.strip() for line in f if line.strip()]
    else:
        # Use ALL videos from the dataset
        _all_ds = PNGRawDataset(
            img_folder=args.base_video_dir,
            gt_folder=args.input_mask_dir,
            file_list_txt=None,
            is_palette=True,
        )
        video_names = list(_all_ds.video_names)
        del _all_ds
        logger.info(f"No video list provided → using all {len(video_names)} videos from dataset")
    logger.info(f"Processing {len(video_names)} videos → {args.output_vis_dir}")

    # Create output directory
    os.makedirs(args.output_vis_dir, exist_ok=True)

    # Process videos sequentially
    for i, video_name in enumerate(video_names):
        logger.info(f"\n[{i+1}/{len(video_names)}] {video_name}")
        try:
            visualize_video(
                model=model,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_vis_dir=args.output_vis_dir,
                video_name=video_name,
                attn_capture=attn_capture,
                num_frames_per_segment=args.num_frames_per_segment,
                transforms=eval_transforms,
                score_thresh=args.score_thresh,
                max_segments=args.max_segments,
                drm_only=args.drm_only,
            )
            logger.info(f"  ✓ Completed successfully")
        except Exception as e:
            logger.error(f"  ✗ Error on {video_name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Cleanup ──
    attn_capture.remove()
    logger.info(f"\nDone! Visualizations saved to {args.output_vis_dir}")


if __name__ == "__main__":
    main()
