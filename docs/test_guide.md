# SAM2 Testing / Inference Guide (Multi-GPU Video Segmentation)

This document describes how to run **inference/testing** for the trained SAM2 training model in this repo (including the **Two-Stage + DAM (RAM/DRM)** variants), using the existing multi-GPU test scripts.

## Table of Contents

1. [What “Testing” Does](#what-testing-does)
2. [Key Scripts](#key-scripts)
3. [Inputs & Expected Folder Structure](#inputs--expected-folder-structure)
4. [How Inference Works (Data Flow)](#how-inference-works-data-flow)
5. [How to Run Multi-GPU Inference](#how-to-run-multi-gpu-inference)
6. [Outputs & Logs](#outputs--logs)
7. [Common Issues](#common-issues)

---

## What “Testing” Does

This test pipeline performs **video segmentation** by always using:

- **Query/conditioning frame**: `00000.jpg` with `00000.png` as the input mask
- **Prediction frames**: the remaining frames of the video, processed in **chunks of N frames** (`--num-frames-per-segment`)
  - Frame 0: query (conditioning)
  - Frames 1–(N-1): consecutive video frames (prediction)

The test code uses the **same training model class** (`training.model.sam2.SAM2Train`) and thus supports:

- **Single-stage** inference (if `enable_two_stage_memory=false`)
- **Two-stage** inference (if `enable_two_stage_memory=true`)
- **DAM (RAM/DRM)** stage-2 memory bank (if `enable_dam=true` and `enable_two_stage_memory=true`)

---

## Key Scripts

- **Multi-GPU driver**: `tools/test_video_segmentation_multi_gpu.sh`
  - Splits the video list across GPUs and launches one inference process per GPU.
  - **Updated behavior in this repo**:
    - Resolves relative `--config` / `--checkpoint` paths automatically
    - Accepts `--checkpoint` as a **file** or **directory** (auto-selects `checkpoint.pt` or newest `*.pt`)
    - Uses unbuffered Python logging so per-GPU logs update live

- **Single-process inference**: `tools/test_video_segmentation.py`
  - Loads the config + checkpoint, iterates videos, and writes predicted PNG masks.

---

## Inputs & Expected Folder Structure

The multi-GPU script expects:

- **Video list file**: a text file containing video names (one per line)
- **Base video dir**: contains subfolders per video, each containing frames like `00000.jpg`, `00001.jpg`, ...
- **Input mask dir**: contains subfolders per video, each containing at least `00000.png` for the query mask

Default paths are set near the top of `tools/test_video_segmentation_multi_gpu.sh` and can be overridden via flags.

---

## How Inference Works (Data Flow)

Per video:

- Load query mask from frame 0 (`00000.png`) and determine object IDs present in the query.
- Split the rest of the video into segments of **N frames total** (`--num-frames-per-segment`):
  - `[query=0] + [i..i+(N-2)]` (padding last frame if needed)
- For each segment:
  - Build a `training.utils.data_utils.VideoDatapoint` where:
    - Frame 0 has the **mask input** for the object(s)
    - Frames 1–(N-1) have **no mask input** (prediction only)
  - Run the model forward (supports stage 1 / stage 2 depending on config)
- Save predicted masks as PNGs per frame into the output directory.

---

## How to Run Multi-GPU Inference

Run from repo root:

```bash
cd <REPO_ROOT>
bash tools/test_video_segmentation_multi_gpu.sh
```

### Example: RAM1_Feb8 Configuration (Latest)

**Background execution with nohup:**

```bash
cd <REPO_ROOT> && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup bash tools/test_video_segmentation_multi_gpu.sh \
  --num-gpus 4 \
  --num-frames-per-segment 8 \
  --video-list-file ./splits/test_t-IoU.txt \
  --output-dir ./test_outputs_2stage_DAM_AdaptiveGate_RAM1_Feb8 \
  --config sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_RAM1_Feb8.yaml \
  --checkpoint sam2_logs/two_stage_mem_dam_adaptive_gating_ram1_Feb8/checkpoints \
  --base-video-dir ./data/VQS-4K/images \
  --input-mask-dir ./data/VQS-4K/masks \
  > inference_RAM1_Feb8.log 2>&1 &
echo "PID: $!"
```

**Direct execution (foreground):**

```bash
bash tools/test_video_segmentation_multi_gpu.sh \
  --num-gpus 4 \
  --num-frames-per-segment 8 \
  --video-list-file ./splits/test_t-IoU.txt \
  --output-dir ./test_outputs_2stage_DAM_AdaptiveGate_RAM1_Feb8 \
  --config sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_RAM1_Feb8.yaml \
  --checkpoint sam2_logs/two_stage_mem_dam_adaptive_gating_ram1_Feb8/checkpoints \
  --base-video-dir ./data/VQS-4K/images \
  --input-mask-dir ./data/VQS-4K/masks \
  --segments-per-batch 4 \
  --score-thresh 0.5
```

### Configuration Notes:

- **Checkpoint argument**:
  - You can pass a **directory** (`.../checkpoints/`) and the script will pick `checkpoint.pt` (or newest `*.pt`).
  - Or pass a specific `.../checkpoint.pt`.
- **Config matters**: inference behavior (two-stage, DAM, adaptive gating) is controlled entirely by the YAML config you pass.
- **Frame count must match training**:
  - If you trained with `scratch.num_frames: K`, then use `--num-frames-per-segment K` for inference.
  - Example: nf8 training (`K=8`) ⇒ `--num-frames-per-segment 8`.
  - RAM1_Feb8 config uses `num_frames: 8`.

### RAM1_Feb8 Configuration Features:
- **Two-stage memory** with DAM (Distractor-Aware Memory)
- **Adaptive gating** for Query/RAM/DRM control
- **RAM size**: 1, **DRM size**: 1
- **Quality-weighted RAM selection** with recency decay (alpha=0.4, decay=0.2)
- **Score-weighted DRM strength mode**
- **Spatio-temporal transformer** for enhanced frame processing

---

## Outputs & Logs

Output directory (example):

- `test_outputs_2stage_DAM_AdaptiveGate_RAM1_Feb8/`
  - `test_inference_gpu0.log`, `test_inference_gpu1.log`, ...
  - Per-video subfolders:
    - `VIDEO_NAME/00000.png`, `VIDEO_NAME/00001.png`, ...

To monitor background execution:

```bash
# Monitor main log file
tail -f inference_RAM1_Feb8.log

# Monitor per-GPU logs
tail -f test_outputs_2stage_DAM_AdaptiveGate_RAM1_Feb8/test_inference_gpu0.log

# Check process status
ps aux | grep test_video_segmentation
```

---

## Common Issues

- **Logs stay empty**:
  - Ensure the script is using unbuffered python (`python -u`). This repo’s script already does that.

- **Checkpoint not found**:
  - Pass `--checkpoint` pointing to either `.../checkpoints/` or `.../checkpoint.pt`.

- **Wrong checkpoint used**:
  - For the RAM1_Feb8 run (latest), use:
    - `sam2_logs/two_stage_mem_dam_adaptive_gating_ram1_Feb8/checkpoints/checkpoint.pt`
  - For older DAM runs, check the appropriate log directory

- **Background process control**:
  - To check if inference is running:
    ```bash
    ps aux | grep test_video_segmentation
    ```
  - To view the background process ID:
    ```bash
    jobs -l
    ```

- **Stopping inference**:
  - Kill the per-GPU Python workers:

```bash
pkill -f "python tools/test_video_segmentation.py"
```


