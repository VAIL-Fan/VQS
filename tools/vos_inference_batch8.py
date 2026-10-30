# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VOS inference script that processes 8 frames at a time (like training):
- Uses the same dataset infrastructure as training
- Divides video into chunks of 8 frames (1 query + 7 video frames)
- Concatenates all predictions
- Only saves non-empty masks
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
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


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        # Convert to boolean if it's 0/255 mask
        if object_mask.dtype == np.uint8 and np.max(object_mask) == 255:
            object_mask = object_mask > 0
        mask[object_mask] = object_id
    return mask


def is_mask_empty(mask):
    """Check if a mask is empty (all zeros)."""
    return np.sum(mask) == 0


class ChunkSampler:
    """Sampler that divides video into chunks of 8 frames for inference."""
    
    def __init__(self, num_frames=8):
        self.num_frames = num_frames
    
    def sample_chunks(self, video, segment_loader):
        """Divide video into chunks of 8 frames (1 query + 7 consecutive)."""
        chunks = []
        num_video_frames = len(video.frames)
        
        # Get object IDs from first frame
        first_frame_segments = segment_loader.load(video.frames[0].frame_idx)
        object_ids = sorted([obj_id for obj_id in first_frame_segments.keys() if first_frame_segments[obj_id].sum() > 0])
        
        if len(object_ids) == 0:
            return chunks, object_ids
        
        # Process video in chunks
        # Each chunk: frame 0 is query (first frame), frames 1-7 are consecutive video frames
        start_idx = 0
        while start_idx < num_video_frames:
            # Query frame is always the first frame of the video (or chunk start)
            query_frame_idx = 0 if start_idx == 0 else start_idx
            
            # Get 7 consecutive frames starting from start_idx + 1
            video_frame_start = start_idx + 1
            video_frame_end = min(video_frame_start + (self.num_frames - 1), num_video_frames)
            
            if video_frame_end - video_frame_start < (self.num_frames - 1):
                # Not enough frames, pad with last frame
                video_frame_indices = list(range(video_frame_start, video_frame_end))
                while len(video_frame_indices) < (self.num_frames - 1):
                    video_frame_indices.append(video_frame_end - 1 if video_frame_end > video_frame_start else start_idx)
            else:
                video_frame_indices = list(range(video_frame_start, video_frame_end))
            
            # Create chunk: query frame + 7 video frames
            chunk_frames = [video.frames[query_frame_idx]] + [video.frames[i] for i in video_frame_indices]
            
            chunks.append(chunk_frames)
            
            # Move to next chunk (overlap by 1 frame: last video frame becomes next query)
            start_idx = video_frame_indices[-1] if video_frame_indices else start_idx + self.num_frames
            
            # Break if we've processed all frames
            if start_idx >= num_video_frames - 1:
                break
        
        return chunks, object_ids


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference_batch8(
    model,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    num_frames_per_batch=8,
    transforms=None,
):
    """
    Run VOS inference processing 8 frames at a time (like training).
    Uses the same dataset infrastructure as training.
    """
    device = next(model.parameters()).device
    
    # Create raw dataset (same as training)
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
    chunk_sampler = ChunkSampler(num_frames=num_frames_per_batch)
    chunks, object_ids = chunk_sampler.sample_chunks(video, segment_loader)
    
    if len(chunks) == 0:
        print(f"  No valid chunks for video {video_name}")
        return
    
    print(f"  Processing {len(chunks)} chunks with {len(object_ids)} objects")
    
    # Store all predictions: {frame_idx: {obj_id: mask}}
    all_predictions = defaultdict(dict)
    
    # Get original image size from first frame
    first_frame_path = video.frames[0].image_path
    first_img = Image.open(first_frame_path).convert("RGB")
    orig_height, orig_width = first_img.size[1], first_img.size[0]
    
    # Get palette from first frame mask
    first_mask_path = os.path.join(input_mask_dir, video_name, f"{video.frames[0].frame_idx}.png")
    if os.path.exists(first_mask_path):
        _, input_palette = load_ann_png(first_mask_path)
    else:
        input_palette = DAVIS_PALETTE
    output_palette = input_palette or DAVIS_PALETTE
    
    # Process each chunk
    for chunk_idx, chunk_frames in enumerate(chunks):
        print(f"    Chunk {chunk_idx + 1}/{len(chunks)}: frames {[f.frame_idx for f in chunk_frames]}")
        
        # Create VideoDatapoint using training infrastructure
        rgb_images = load_images(chunk_frames)
        images = []
        for frame_idx, frame in enumerate(chunk_frames):
            w, h = rgb_images[frame_idx].size
            images.append(Frame(data=rgb_images[frame_idx], objects=[]))
            
            # Load segments for this frame
            segments = segment_loader.load(frame.frame_idx)
            
            # Add objects (only for query frame, others have empty masks)
            for obj_id in object_ids:
                if frame_idx == 0:  # Query frame
                    if obj_id in segments:
                        segment = segments[obj_id].to(torch.uint8)
                    else:
                        segment = torch.zeros(h, w, dtype=torch.uint8)
                else:  # Video frames - no mask input
                    segment = torch.zeros(h, w, dtype=torch.uint8)
                
                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )
        
        video_datapoint = VideoDatapoint(
            frames=images,
            video_id=0,
            size=(h, w),
        )
        
        # Apply transforms if provided (same as training)
        if transforms:
            video_datapoint = transforms(video_datapoint, epoch=0)
        
        # Collate into batch (same as training)
        batched_datapoint = collate_fn([video_datapoint], dict_key="all")
        batched_datapoint = batched_datapoint.to(device)
        
        # Forward pass (same as training)
        flat_img_batch = batched_datapoint.flat_img_batch
        backbone_out = model.forward_image(flat_img_batch)
        
        # Prepare prompt inputs (same as training)
        model.prepare_prompt_inputs(backbone_out, batched_datapoint, start_frame_idx=0)
        
        # Override to use only frame 0 as conditioning frame with mask input
        backbone_out["init_cond_frames"] = [0]
        backbone_out["frames_to_add_correction_pt"] = []
        backbone_out["use_pt_input"] = False
        
        # Set mask input for frame 0 (combine all object masks)
        if 0 in backbone_out["gt_masks_per_frame"]:
            query_masks = backbone_out["gt_masks_per_frame"][0]  # [num_objs, 1, H, W]
            query_mask_combined = query_masks.any(dim=0, keepdim=True).float()  # [1, 1, H, W]
            backbone_out["mask_inputs_per_frame"] = {0: query_mask_combined}
        
        # Forward tracking
        outputs = model.forward_tracking(backbone_out, batched_datapoint, return_dict=False)
        
        # Extract predictions
        for i, frame in enumerate(chunk_frames):
            frame_idx = frame.frame_idx
            output = outputs[i]
            
            # Use high-resolution masks if available, otherwise fall back to low-res
            if "pred_masks_high_res" in output:
                pred_masks = output["pred_masks_high_res"]  # [num_objs, H, W] - high res (1024x1024)
            else:
                pred_masks = output["pred_masks"]  # [num_objs, H, W] - low res (256x256)
            
            for obj_idx, obj_id in enumerate(object_ids):
                if obj_idx < pred_masks.shape[0]:
                    mask_logits = pred_masks[obj_idx].cpu().numpy()  # [H, W] - these are logits
                    
                    # Ensure 2D mask
                    if mask_logits.ndim > 2:
                        mask_logits = mask_logits.squeeze()
                    if mask_logits.ndim != 2:
                        continue
                    
                    # Apply sigmoid to convert logits to probabilities, then threshold
                    mask_probs = torch.sigmoid(torch.from_numpy(mask_logits)).numpy()
                    mask = (mask_probs > 0.5).astype(np.uint8) * 255
                    
                    # Resize to original image size if needed
                    if mask.shape != (orig_height, orig_width):
                        mask_pil = Image.fromarray(mask)
                        mask_pil = mask_pil.resize((orig_width, orig_height), Image.NEAREST)
                        mask = np.array(mask_pil)
                    
                    # Store prediction (may overwrite if frame appears in multiple chunks)
                    all_predictions[frame_idx][obj_id] = mask
    
    # Save predictions (only non-empty masks)
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    for frame_idx, per_obj_mask in all_predictions.items():
        # Filter out empty masks
        non_empty_masks = {
            obj_id: mask for obj_id, mask in per_obj_mask.items()
            if not is_mask_empty(mask)
        }
        
        if non_empty_masks:  # Only save if there are non-empty masks
            combined_mask = put_per_obj_mask(non_empty_masks, orig_height, orig_width)
            frame_name = str(frame_idx)
            output_path = os.path.join(output_mask_dir, video_name, f"{frame_name}.png")
            save_ann_png(output_path, combined_mask, output_palette)


def main():
    parser = argparse.ArgumentParser()
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
        help="directory containing videos (as JPEG files)",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=True,
        help="directory containing input masks (as PNG files)",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    args = parser.parse_args()
    
    # Load model
    GlobalHydra.instance().clear()
    
    with initialize_config_module("sam2", version_base="1.2"):
        cfg = compose(config_name=args.sam2_cfg)
    
    model = instantiate(cfg.trainer.model, _recursive_=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.sam2_checkpoint, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create eval transforms (no augmentation, just resize, tensor, normalize)
    from training.dataset.transforms import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    
    eval_transforms = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=cfg.scratch.resolution, square=True, consistent_transform=True),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transforms = eval_transforms
    
    # Get video list
    if args.video_list_file:
        with open(args.video_list_file, "r") as f:
            video_names = [line.strip() for line in f if line.strip()]
    else:
        video_names = [
            d for d in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, d))
        ]
    
    print(f"Running inference on {len(video_names)} videos")
    for i, video_name in enumerate(video_names):
        print(f"{i+1}/{len(video_names)} - Processing {video_name}")
        try:
            vos_inference_batch8(
                model=model,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                transforms=transforms,
            )
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Completed! Outputs saved to {args.output_mask_dir}")


if __name__ == "__main__":
    main()
