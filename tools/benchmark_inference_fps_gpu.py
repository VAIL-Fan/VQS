#!/usr/bin/env python3
"""
Benchmark script for measuring inference-only FPS (eFPS) and GPU usage.

This script measures:
- Pure GPU inference time (model forward pass only, excludes data loading)
- Effective FPS (eFPS) = total predicted frames / GPU processing time
- GPU memory usage (allocated, reserved, peak)

Uses one video at different resolutions (360p, 480p, 720p) with 8-frame sequences.
Does NOT modify the existing test_video_segmentation.py - this is a standalone benchmark.

Reference: train_guide.md, test_guide.md
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from training.dataset.vos_raw_dataset import PNGRawDataset
from training.dataset.vos_segment_loader import PalettisedPNGSegmentLoader
from training.utils.data_utils import collate_fn, Frame, Object, VideoDatapoint
from training.model.sam2 import SAM2Train
from training.dataset.transforms import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    return {object_id: (mask == object_id) for object_id in object_ids}


class TestChunkSampler:
    """Sampler that divides video into chunks for testing."""

    def __init__(self, num_frames=8):
        self.num_frames = num_frames

    def sample_chunks(self, video, segment_loader):
        chunks = []
        num_video_frames = len(video.frames)
        if num_video_frames < 2:
            return chunks, []

        query_frame_idx = 0
        first_frame_segments = segment_loader.load(video.frames[query_frame_idx].frame_idx)
        object_ids = sorted([obj_id for obj_id in first_frame_segments.keys() if first_frame_segments[obj_id].sum() > 0])
        if len(object_ids) == 0:
            return chunks, object_ids

        query_frame = video.frames[query_frame_idx]
        video_frame_start = 1

        while video_frame_start < num_video_frames:
            video_frame_end = min(video_frame_start + (self.num_frames - 1), num_video_frames)
            video_frame_indices = list(range(video_frame_start, video_frame_end))
            if len(video_frame_indices) < (self.num_frames - 1):
                last_frame_idx = video_frame_indices[-1] if video_frame_indices else video_frame_start
                while len(video_frame_indices) < (self.num_frames - 1):
                    video_frame_indices.append(last_frame_idx)
            chunk_frames = [query_frame] + [video.frames[i] for i in video_frame_indices]
            chunks.append(chunk_frames)
            video_frame_start = video_frame_end

        return chunks, object_ids


def load_ann_png(path):
    mask = Image.open(path)
    return np.array(mask).astype(np.uint8)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_benchmark(
    model,
    base_video_dir,
    input_mask_dir,
    video_name,
    resolution,
    num_frames_per_segment=8,
    segments_per_batch=1,
    num_warmup=2,
    num_timing_runs=5,
):
    """
    Run inference benchmark for one video at given resolution.
    Returns: (gpu_time_sec, total_predicted_frames, gpu_mem_allocated_mb, gpu_mem_reserved_mb, gpu_mem_peak_mb)
    """
    device = next(model.parameters()).device

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
        raise RuntimeError(f"Video {video_name} not found")

    video, segment_loader = video_dataset.get_video(video_idx)
    chunk_sampler = TestChunkSampler(num_frames=num_frames_per_segment)
    chunks, object_ids = chunk_sampler.sample_chunks(video, segment_loader)

    if len(chunks) == 0:
        raise RuntimeError(f"No valid chunks for video {video_name}")

    transforms = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=(resolution,), square=True, consistent_transform=True),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_cache = {}
    for frame in video.frames:
        frame_img = Image.open(frame.image_path).convert("RGB")
        frame_cache[frame.frame_idx] = frame_img

    query_frame_0_segments = segment_loader.load(0)

    def build_datapoint_from_chunk(chunk_frames):
        rgb_images = [frame_cache[f.frame_idx] for f in chunk_frames]
        images = []
        size_hw = None
        for local_idx, frame in enumerate(chunk_frames):
            w, h = rgb_images[local_idx].size
            size_hw = (h, w)
            images.append(Frame(data=rgb_images[local_idx], objects=[]))
            segments = query_frame_0_segments if local_idx == 0 else segment_loader.load(frame.frame_idx)
            for obj_id in object_ids:
                if local_idx == 0 and obj_id in segments:
                    segment = segments[obj_id].to(torch.uint8)
                elif local_idx == 0:
                    segment = torch.zeros(h, w, dtype=torch.uint8)
                else:
                    segment = torch.zeros(h, w, dtype=torch.uint8)
                images[local_idx].objects.append(Object(object_id=obj_id, frame_index=frame.frame_idx, segment=segment))
        return VideoDatapoint(frames=images, video_id=0, size=size_hw), [f.frame_idx for f in chunk_frames]

    total_predicted_frames = 0
    for chunk_frames in chunks:
        total_predicted_frames += len(chunk_frames) - 1  # exclude query frame

    # Warmup runs
    for _ in range(num_warmup):
        batch_chunks = chunks[:segments_per_batch]
        batch_infos = []
        for chunk_frames in batch_chunks:
            video_datapoint, frame_indices = build_datapoint_from_chunk(chunk_frames)
            video_datapoint = transforms(video_datapoint, epoch=0)
            batch_infos.append({"frame_indices": frame_indices, "datapoint": video_datapoint})

        datapoints = [info["datapoint"] for info in batch_infos]
        batched_datapoint = collate_fn(datapoints, dict_key="all").to(device)
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
                    zeros = torch.zeros((1, 1, query_masks.shape[-2], query_masks.shape[-1]), dtype=query_masks.dtype, device=query_masks.device).float()
                    batch_mask_list.append(zeros)
                else:
                    batch_mask_list.append(vid_mask.any(dim=0, keepdim=True).float())
            if batch_mask_list:
                backbone_out["mask_inputs_per_frame"] = {0: torch.cat(batch_mask_list, dim=0)}

        if getattr(model, "enable_two_stage_memory", False):
            stage1_outputs = model.forward_tracking(backbone_out, batched_datapoint, return_dict=True)
            _ = model._run_second_stage(backbone_out, batched_datapoint, stage1_outputs)
        else:
            _ = model.forward_tracking(backbone_out, batched_datapoint, return_dict=False)

        torch.cuda.synchronize()

    # Timing runs with CUDA events
    gpu_times = []
    for _ in range(num_timing_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for batch_start in range(0, len(chunks), segments_per_batch):
            batch_chunks = chunks[batch_start:batch_start + segments_per_batch]
            batch_infos = []
            for chunk_frames in batch_chunks:
                video_datapoint, frame_indices = build_datapoint_from_chunk(chunk_frames)
                video_datapoint = transforms(video_datapoint, epoch=0)
                batch_infos.append({"frame_indices": frame_indices, "datapoint": video_datapoint})

            datapoints = [info["datapoint"] for info in batch_infos]
            batched_datapoint = collate_fn(datapoints, dict_key="all").to(device)
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
                        zeros = torch.zeros((1, 1, query_masks.shape[-2], query_masks.shape[-1]), dtype=query_masks.dtype, device=query_masks.device).float()
                        batch_mask_list.append(zeros)
                    else:
                        batch_mask_list.append(vid_mask.any(dim=0, keepdim=True).float())
                if batch_mask_list:
                    backbone_out["mask_inputs_per_frame"] = {0: torch.cat(batch_mask_list, dim=0)}

            if getattr(model, "enable_two_stage_memory", False):
                stage1_outputs = model.forward_tracking(backbone_out, batched_datapoint, return_dict=True)
                _ = model._run_second_stage(backbone_out, batched_datapoint, stage1_outputs)
            else:
                _ = model.forward_tracking(backbone_out, batched_datapoint, return_dict=False)

        end_event.record()
        torch.cuda.synchronize()
        gpu_times.append(start_event.elapsed_time(end_event) / 1000.0)

    gpu_time_sec = np.median(gpu_times)
    gpu_mem_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    gpu_mem_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    gpu_mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0

    return gpu_time_sec, total_predicted_frames, gpu_mem_allocated_mb, gpu_mem_reserved_mb, gpu_mem_peak_mb


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference FPS and GPU usage")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--base_video_dir", type=str, required=True, help="Base video directory")
    parser.add_argument("--input_mask_dir", type=str, required=True, help="Input mask directory")
    parser.add_argument("--video_name", type=str, default="aircraft-airship-set_4", help="Single video to benchmark")
    parser.add_argument("--num_frames_per_segment", type=int, default=8, help="Frames per segment (1 query + N-1 video)")
    parser.add_argument("--segments_per_batch", type=int, default=1, help="Segments per batch")
    parser.add_argument("--num_warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--num_timing_runs", type=int, default=5, help="Timing runs for median")
    parser.add_argument("--output_md", type=str, default="INFERENCE_FPS_GPU_BENCHMARK.md", help="Output markdown file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = instantiate(cfg.trainer.model, _recursive_=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Model is trained at config resolution (typically 768). Use that for benchmarking.
    # 360p/480p would require models trained at those resolutions (architecture is resolution-dependent).
    resolution = getattr(cfg.scratch, "resolution", 768)
    res_label = "720p" if resolution >= 720 else f"{resolution}p"
    resolutions = [(resolution, res_label)]

    results = []
    print(f"\nBenchmarking on video: {args.video_name}")
    print(f"Num frames per segment: {args.num_frames_per_segment} (1 query + {args.num_frames_per_segment - 1} video)")
    print(f"Resolution: {resolution}x{resolution} (model native resolution)")
    print("-" * 60)

    for res_val, res_name in resolutions:
        print(f"\nResolution: {res_name} ({res_val}x{res_val})")
        try:
            gpu_time_sec, total_frames, mem_alloc, mem_reserved, mem_peak = run_benchmark(
                model=model,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                video_name=args.video_name,
                resolution=res_val,
                num_frames_per_segment=args.num_frames_per_segment,
                segments_per_batch=args.segments_per_batch,
                num_warmup=args.num_warmup,
                num_timing_runs=args.num_timing_runs,
            )
            efps = total_frames / gpu_time_sec if gpu_time_sec > 0 else 0
            results.append({
                "resolution": res_name,
                "res_val": res_val,
                "gpu_time_sec": gpu_time_sec,
                "total_frames": total_frames,
                "efps": efps,
                "mem_alloc_mb": mem_alloc,
                "mem_reserved_mb": mem_reserved,
                "mem_peak_mb": mem_peak,
            })
            print(f"  GPU time: {gpu_time_sec:.3f} s")
            print(f"  Total predicted frames: {total_frames}")
            print(f"  eFPS: {efps:.2f}")
            print(f"  GPU mem allocated: {mem_alloc:.1f} MB")
            print(f"  GPU mem reserved: {mem_reserved:.1f} MB")
            print(f"  GPU mem peak: {mem_peak:.1f} MB")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "resolution": res_name,
                "res_val": res_val,
                "error": str(e),
            })

    # Write results to MD file
    output_path = args.output_md
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    with open(output_path, "w") as f:
        f.write("# Inference FPS and GPU Usage Benchmark\n\n")
        f.write("Measured inference-only eFPS (effective FPS) and GPU memory usage.\n")
        f.write("Excludes data loading; measures pure GPU forward pass time.\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Video**: {args.video_name}\n")
        f.write(f"- **Frames per segment**: {args.num_frames_per_segment} (1 query + {args.num_frames_per_segment - 1} video)\n")
        f.write(f"- **Config**: {args.config}\n")
        f.write(f"- **Checkpoint**: {args.checkpoint}\n")
        f.write(f"- **GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n\n")
        f.write("## Results\n\n")
        f.write("| Resolution | eFPS | GPU Time (s) | Predicted Frames | GPU Mem Alloc (MB) | GPU Mem Peak (MB) |\n")
        f.write("|------------|------|--------------|------------------|--------------------|--------------------|\n")
        for r in results:
            if "error" in r:
                f.write(f"| {r['resolution']} | ERROR | - | - | - | - |\n")
            else:
                f.write(f"| {r['resolution']} ({r['res_val']}px) | {r['efps']:.2f} | {r['gpu_time_sec']:.3f} | {r['total_frames']} | {r['mem_alloc_mb']:.1f} | {r['mem_peak_mb']:.1f} |\n")
        f.write("\n## Notes\n\n")
        f.write("- eFPS = total predicted frames / GPU processing time (inference-only, no data loader)\n")
        f.write("- This model is trained at a fixed resolution; only the native resolution is benchmarked.\n")
        f.write("- For 360p/480p, models trained at those resolutions would be needed (architecture is resolution-dependent).\n")
        f.write("- Run with: `python tools/benchmark_inference_fps_gpu.py --config ... --checkpoint ... --base_video_dir ... --input_mask_dir ...`\n")

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
