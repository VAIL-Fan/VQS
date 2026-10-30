# Inference FPS and GPU Usage Benchmark

Measured inference-only eFPS (effective FPS) and GPU memory usage.
Excludes data loading; measures pure GPU forward pass time.

## Configuration

- **Video**: aircraft-airship-set_4
- **Frames per segment**: 8 (1 query + 7 video)
- **Config**: sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml
- **Checkpoint**: sam2_logs/two_stage_mem_dam_adaptive_gating_Jan28/checkpoints/checkpoint.pt
- **GPU**: NVIDIA RTX A6000

## Measured Results (720p / 768px)

| Resolution | eFPS | GPU Time (s) | Predicted Frames | GPU Mem Alloc (MB) | GPU Mem Peak (MB) |
|------------|------|--------------|------------------|--------------------|--------------------|
| 720p (768px) | **29.43** | 12.368 | 364 | 1563.8 | 2519.6 |

## Efficiency Table (8-frame sequences)

| Resolution | eFPS | GPU Mem Peak | Notes |
|------------|------|--------------|-------|
| 360p (384px) | ~70 (est.) | - | Model not trained at this resolution; scaled from 720p |
| 480p (512px) | ~45 (est.) | - | Model not trained at this resolution; scaled from 720p |
| 720p (768px) | **29.43** | 2520 MB | Measured on aircraft-airship-set_4, NVIDIA RTX A6000 |

*Estimates for 360p/480p assume compute scales ~ resolution². Actual values require models trained at those resolutions.*

## Notes

- **eFPS** = total predicted frames / GPU processing time (inference-only, no data loader)
- This model is trained at 768×768; only the native resolution is benchmarked.
- For 360p/480p, models trained at those resolutions would be needed (architecture is resolution-dependent).
- Run with: `python tools/benchmark_inference_fps_gpu.py --config ... --checkpoint ... --base_video_dir ... --input_mask_dir ...`
