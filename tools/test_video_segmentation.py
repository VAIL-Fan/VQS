# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for video segmentation:
- Always uses frame 00000.jpg and mask 00000.png as query (conditioning frame)
- Splits remaining frames into segments of N frames total (configurable via --num_frames_per_segment):
  * Frame 0: Query frame (00000) - conditioning frame with mask input
  * Frames 1..N-1: consecutive video frames - prediction frames (no mask input)
- Processes each segment using the training model (set N to match training clip length)
- Combines all segment predictions to get final masks for all frames
- Saves masks as PNG files in output folder

Note: Default is 16 total frames per segment (1 query + 15 video frames).
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from training.dataset.vos_raw_dataset import PNGRawDataset
from training.dataset.vos_dataset import VOSDataset, load_images
from training.dataset.vos_segment_loader import PalettisedPNGSegmentLoader
from training.utils.data_utils import collate_fn, Frame, Object, VideoDatapoint
from training.model.sam2 import SAM2Train

# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette=None):
    """Save a mask as a PNG file in grayscale mode (L) with 0 and 255 values.
    
    Always saves as grayscale mode regardless of palette parameter.
    """
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    
    # Always save as grayscale (L mode) with 0 and 255 values
    output_mask = Image.fromarray(mask)
    output_mask = output_mask.convert('L')
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single binary mask (0 for background, 255 for foreground)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        # Convert to boolean if it's 0/255 mask
        if object_mask.dtype == np.uint8 and np.max(object_mask) == 255:
            object_mask = object_mask > 0
        # Set any non-zero pixel to 255 (binary format)
        mask[object_mask] = 255
    return mask


def is_mask_empty(mask):
    """Check if a mask is empty (all zeros)."""
    return np.sum(mask) == 0


class TestChunkSampler:
    """Sampler that divides video into chunks for testing.
    
    Always uses frame 00000 as query frame (conditioning frame).
    Splits remaining frames into segments of `num_frames` total:
    - Frame 0: Query frame (00000) - conditioning frame with mask
    - Frames 1..num_frames-1: consecutive video frames - prediction frames
    
    Set `num_frames` to match training clip length (see `scratch.num_frames` in the training YAML).
    """
    
    def __init__(self, num_frames=16):
        # Total frames per segment: 1 query + (num_frames - 1) video frames
        self.num_frames = num_frames
    
    def sample_chunks(self, video, segment_loader):
        """Divide video into chunks: always use frame 0 as query, split remaining frames into segments."""
        chunks = []
        num_video_frames = len(video.frames)
        
        if num_video_frames < 2:
            return chunks, []
        
        # Get object IDs from frame 0 (query frame)
        query_frame_idx = 0
        first_frame_segments = segment_loader.load(video.frames[query_frame_idx].frame_idx)
        object_ids = sorted([obj_id for obj_id in first_frame_segments.keys() if first_frame_segments[obj_id].sum() > 0])
        
        if len(object_ids) == 0:
            return chunks, object_ids
        
        # Query frame is always frame 0 (00000)
        query_frame = video.frames[query_frame_idx]
        
        # Split remaining frames (frames 1 onwards) into segments
        # Each segment has `self.num_frames` total:
        # [frame_0 (query), frame_i, frame_i+1, ..., frame_i+(self.num_frames-2)]
        # Frame 0: Query/conditioning frame (has mask input)
        # Frames 1..num_frames-1: consecutive video frames (prediction frames, no mask input)
        video_frame_start = 1  # Start from frame 1 (frame 0 is query)
        
        while video_frame_start < num_video_frames:
            # Get (self.num_frames - 1) consecutive video frames (prediction frames)
            video_frame_end = min(video_frame_start + (self.num_frames - 1), num_video_frames)
            video_frame_indices = list(range(video_frame_start, video_frame_end))
            
            # If we don't have enough frames, pad with the last frame
            if len(video_frame_indices) < (self.num_frames - 1):
                last_frame_idx = video_frame_indices[-1] if video_frame_indices else video_frame_start
                while len(video_frame_indices) < (self.num_frames - 1):
                    video_frame_indices.append(last_frame_idx)
            
            # Create chunk: `self.num_frames` total = query frame (frame 0) + (self.num_frames - 1) video frames
            chunk_frames = [query_frame] + [video.frames[i] for i in video_frame_indices]
            chunks.append(chunk_frames)
            
            # Move to next segment
            video_frame_start = video_frame_end
        
        return chunks, object_ids


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def test_video_segmentation(
    model,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    num_frames_per_segment=16,
    segments_per_batch=1,
    transforms=None,
):
    """
    Test video segmentation:
    - Always uses frame 00000 as query (conditioning frame with mask)
    - Splits remaining frames into segments of `num_frames_per_segment` total:
      * Frame 0: Query frame (00000) - conditioning frame
      * Frames 1..N-1: consecutive video frames - prediction frames
    - Processes each segment using the training model (set N to match training clip length)
    - Combines predictions from all segments
    """
    device = next(model.parameters()).device
    
    # Create raw dataset
    video_dataset = PNGRawDataset(
        img_folder=base_video_dir,
        gt_folder=input_mask_dir,
        file_list_txt=None,  # We'll process single video
        is_palette=True,
    )
    
    # Find video index
    video_idx = None
    for idx, name in enumerate(video_dataset.video_names):
        if name == video_name:
            video_idx = idx
            break
    
    if video_idx is None:
        raise RuntimeError(f"Video {video_name} not found in dataset")
    
    # Get video and segment loader
    video, segment_loader = video_dataset.get_video(video_idx)
    
    # Create chunk sampler
    chunk_sampler = TestChunkSampler(num_frames=num_frames_per_segment)
    chunks, object_ids = chunk_sampler.sample_chunks(video, segment_loader)
    
    if len(chunks) == 0:
        print(f"  No valid chunks for video {video_name}")
        return
    
    print(f"  Processing {len(chunks)} segments with {len(object_ids)} objects")
    print(f"  Total frames in video: {len(video.frames)}")
    
    # Store all predictions: {frame_idx: {obj_id: mask}}
    all_predictions = defaultdict(dict)
    
    # OPTIMIZATION: Preload all frames once per video to avoid repeated I/O
    # Cache frame images and sizes
    print(f"  Preloading {len(video.frames)} frames...")
    frame_cache = {}
    frame_original_sizes = {}
    for frame in video.frames:
        frame_img = Image.open(frame.image_path).convert("RGB")
        frame_cache[frame.frame_idx] = frame_img
        frame_original_sizes[frame.frame_idx] = frame_img.size  # (width, height)
    
    # Get original image size from first frame (for reference)
    orig_width, orig_height = frame_original_sizes[video.frames[0].frame_idx]
    
    # Cache frame 0 (query frame) since it's used in every segment
    query_frame_0 = frame_cache[0]
    query_frame_0_segments = segment_loader.load(0)
    
    # Always save output masks as grayscale (L mode) with 0 and 255 values
    # No need to check input format - output will always be grayscale
    
    segments_per_batch = max(1, segments_per_batch)

    def build_datapoint_from_chunk(chunk_frames):
        # OPTIMIZATION: Use cached frames instead of loading from disk
        rgb_images = []
        for frame in chunk_frames:
            rgb_images.append(frame_cache[frame.frame_idx])
        
        images = []
        size_hw = None
        for local_idx, frame in enumerate(chunk_frames):
            w, h = rgb_images[local_idx].size
            size_hw = (h, w)
            images.append(Frame(data=rgb_images[local_idx], objects=[]))
            
            # OPTIMIZATION: For frame 0, use cached segments
            if local_idx == 0:
                segments = query_frame_0_segments
            else:
                segments = segment_loader.load(frame.frame_idx)
            
            for obj_id in object_ids:
                if local_idx == 0 and obj_id in segments:
                    segment = segments[obj_id].to(torch.uint8)
                elif local_idx == 0:
                    segment = torch.zeros(h, w, dtype=torch.uint8)
                else:
                    segment = torch.zeros(h, w, dtype=torch.uint8)
                images[local_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )
        return VideoDatapoint(frames=images, video_id=0, size=size_hw), [f.frame_idx for f in chunk_frames]

    total_segments = len(chunks)
    for batch_start in range(0, total_segments, segments_per_batch):
        batch_chunks = chunks[batch_start:batch_start + segments_per_batch]
        batch_infos = []

        for local_idx, chunk_frames in enumerate(batch_chunks):
            segment_number = batch_start + local_idx + 1
            frame_indices = [f.frame_idx for f in chunk_frames]
            print(f"    Segment {segment_number}/{len(chunks)}: query={frame_indices[0]}, video_frames={frame_indices[1:]}")
            video_datapoint, frame_indices = build_datapoint_from_chunk(chunk_frames)
            if transforms:
                video_datapoint = transforms(video_datapoint, epoch=0)
            batch_infos.append(
                {
                    "frame_indices": frame_indices,
                    "datapoint": video_datapoint,
                }
            )

        if not batch_infos:
            continue

        datapoints = [info["datapoint"] for info in batch_infos]
        batched_datapoint = collate_fn(datapoints, dict_key="all")
        batched_datapoint = batched_datapoint.to(device)

        flat_img_batch = batched_datapoint.flat_img_batch
        backbone_out = model.forward_image(flat_img_batch)

        model.prepare_prompt_inputs(backbone_out, batched_datapoint, start_frame_idx=0)
        backbone_out["init_cond_frames"] = [0]
        backbone_out["frames_to_add_correction_pt"] = []
        backbone_out["use_pt_input"] = False

        if 0 in backbone_out["gt_masks_per_frame"]:
            query_masks = backbone_out["gt_masks_per_frame"][0]
            obj_to_frame_idx = batched_datapoint.obj_to_frame_idx[0]
            video_indices = obj_to_frame_idx[:, 1].to(torch.long)
            batch_mask_list = []
            for vid_idx in range(batched_datapoint.img_batch.shape[1]):
                vid_mask = query_masks[video_indices == vid_idx]
                if vid_mask.numel() == 0:
                    zeros = torch.zeros(
                        (1, 1, query_masks.shape[-2], query_masks.shape[-1]),
                        dtype=query_masks.dtype,
                        device=query_masks.device,
                    ).float()
                    batch_mask_list.append(zeros)
                else:
                    batch_mask_list.append(vid_mask.any(dim=0, keepdim=True).float())
            if batch_mask_list:
                combined_masks = torch.cat(batch_mask_list, dim=0)
                backbone_out["mask_inputs_per_frame"] = {0: combined_masks}

        # Support for Two-Stage Memory Refinement
        if getattr(model, "enable_two_stage_memory", False):
            # Stage 1
            stage1_outputs = model.forward_tracking(
                backbone_out, 
                batched_datapoint, 
                return_dict=True,
                # Ensure caching if needed, though run_second_stage handles it via backbone_out/cache
            )
            # Stage 2
            # The _run_second_stage method expects stage1_outputs as a dictionary
            outputs = model._run_second_stage(backbone_out, batched_datapoint, stage1_outputs)
        else:
            # Standard single stage
            outputs = model.forward_tracking(backbone_out, batched_datapoint, return_dict=False)

        for frame_pos, output in enumerate(outputs):
            if "pred_masks_high_res" in output:
                pred_masks = output["pred_masks_high_res"]
            else:
                pred_masks = output.get("pred_masks")
            if pred_masks is None:
                continue
            if pred_masks.dim() == 3:
                pred_masks = pred_masks.unsqueeze(0)

            for batch_idx, info in enumerate(batch_infos):
                if frame_pos >= len(info["frame_indices"]):
                    continue
                frame_idx = info["frame_indices"][frame_pos]
                per_frame_pred = pred_masks[batch_idx]
                for obj_idx, obj_id in enumerate(object_ids):
                    if obj_idx >= per_frame_pred.shape[0]:
                        break
                    mask_logits = per_frame_pred[obj_idx]  # [H, W] on GPU
                    
                    # OPTIMIZATION: Keep sigmoid and threshold on GPU
                    mask_probs = torch.sigmoid(mask_logits)  # [H, W] on GPU
                    mask_binary = (mask_probs > score_thresh).float()  # [H, W] on GPU, 0.0 or 1.0
                    
                    # Get target size
                    orig_size = frame_original_sizes.get(frame_idx, (orig_width, orig_height))
                    orig_w, orig_h = orig_size
                    
                    # OPTIMIZATION: Resize on GPU if possible, otherwise on CPU
                    if mask_binary.shape != (orig_h, orig_w):
                        # Resize on GPU using interpolation
                        mask_binary = mask_binary.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                        mask_binary = F.interpolate(
                            mask_binary, 
                            size=(orig_h, orig_w), 
                            mode='nearest'
                        ).squeeze(0).squeeze(0)  # [orig_h, orig_w]
                    
                    # Move to CPU only at the end and convert to uint8
                    mask = (mask_binary.cpu().numpy() * 255).astype(np.uint8)

                    if frame_idx not in all_predictions or frame_idx == 0:
                        all_predictions[frame_idx][obj_id] = mask
    
    # Create output directory
    output_video_dir = os.path.join(output_mask_dir, video_name)
    os.makedirs(output_video_dir, exist_ok=True)
    
    # Process all frames and collect non-empty masks
    frames_with_predictions = []
    for frame in video.frames:
        frame_idx = frame.frame_idx
        # Get original size for this frame
        orig_size = frame_original_sizes.get(frame_idx, (orig_width, orig_height))
        orig_w, orig_h = orig_size  # PIL format: (width, height)
        
        if frame_idx in all_predictions:
            per_obj_mask = all_predictions[frame_idx]
            # Filter out empty masks - only keep non-empty ones
            non_empty_masks = {
                obj_id: mask for obj_id, mask in per_obj_mask.items()
                if not is_mask_empty(mask)
            }
            
            # Only save if there are non-empty masks
            if non_empty_masks:
                combined_mask = put_per_obj_mask(non_empty_masks, orig_h, orig_w)
                # Convert to binary format: 0 for background, 255 for foreground
                # (any non-zero pixel becomes 255)
                binary_mask = (combined_mask > 0).astype(np.uint8) * 255
                
                # Ensure frame_idx is an integer and format as 5-digit with leading zeros
                frame_idx_int = int(frame_idx)
                frame_name = f"{frame_idx_int:05d}"  # Format as 00000, 00001, etc.
                output_path = os.path.join(output_video_dir, f"{frame_name}.png")
                # Always save as grayscale (palette parameter ignored)
                save_ann_png(output_path, binary_mask, palette=None)
                frames_with_predictions.append(frame_idx)
    
    # If all predictions are empty, create an empty folder to indicate processing is done
    if len(frames_with_predictions) == 0:
        # Just create an empty folder (no marker file) to indicate the video was processed
        print(f"  No non-empty predictions found. Created empty folder: {output_video_dir}")
    else:
        print(f"  Saved {len(frames_with_predictions)} frames with non-empty masks.")


def main():
    parser = argparse.ArgumentParser(description="Test video segmentation with trained model")
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml",
        help="SAM 2 model configuration file (training config)",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        required=True,
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing video frames (JPEG files). Images and masks should be in the same folder structure.",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=True,
        help="directory containing input masks (PNG files). Should have same structure as base_video_dir.",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        required=True,
        help="text file containing the list of video names to test",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--num_frames_per_segment",
        type=int,
        default=16,
        help="Total frames per segment (query + prediction frames). Default: 16 (1 query + 15 video frames).",
    )
    parser.add_argument(
        "--segments_per_batch",
        type=int,
        default=4,
        help="Number of segments (from the same video) to process in parallel on a GPU. Default: 4 (optimized for speed).",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="threshold for the output mask probabilities (default: 0.5)",
    )
    args = parser.parse_args()
    
    # Load model config: support both Hydra config names and direct YAML paths
    if os.path.isfile(args.sam2_cfg):
        cfg = OmegaConf.load(args.sam2_cfg)
    else:
        GlobalHydra.instance().clear()
        with initialize_config_module("sam2", version_base="1.2"):
            cfg = compose(config_name=args.sam2_cfg)
    
    model = instantiate(cfg.trainer.model, _recursive_=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.sam2_checkpoint}")
    checkpoint = torch.load(args.sam2_checkpoint, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Create eval transforms (no augmentation, just resize, tensor, normalize)
    from training.dataset.transforms import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    
    eval_transforms = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=cfg.scratch.resolution, square=True, consistent_transform=True),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transforms = eval_transforms
    
    # Get video list
    with open(args.video_list_file, "r") as f:
        video_names = [line.strip() for line in f if line.strip()]
    
    print(f"Running inference on {len(video_names)} videos")
    print(f"Output directory: {args.output_mask_dir}")
    
    for i, video_name in enumerate(video_names):
        print(f"\n{i+1}/{len(video_names)} - Processing {video_name}")
        try:
            test_video_segmentation(
                model=model,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                num_frames_per_segment=args.num_frames_per_segment,
                segments_per_batch=args.segments_per_batch,
                transforms=transforms,
            )
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nCompleted! Outputs saved to {args.output_mask_dir}")


if __name__ == "__main__":
    main()

