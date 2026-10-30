#!/bin/bash
# Multi-GPU test script for video segmentation
# Splits video list across multiple GPUs for parallel processing

set -e

# Default values
NUM_GPUS=4
NUM_FRAMES_PER_SEGMENT=16  # 1 query frame + 15 video frames (set to match training clip length for ablations)
SEGMENTS_PER_BATCH=4       # Number of segments from the same video processed together per GPU (optimized default)
VIDEO_LIST_FILE="./splits/test_t-IoU.txt"
OUTPUT_DIR="./test_outputs_2stage_DAM_Dec28"
CONFIG_FILE="sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml"
CHECKPOINT="sam2_logs/two_stage_mem_dam/checkpoints/checkpoint.pt"
BASE_VIDEO_DIR="./data/VQS-4K/images"
INPUT_MASK_DIR="./data/VQS-4K/masks"
SCORE_THRESH=0.5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num-frames-per-segment)
            NUM_FRAMES_PER_SEGMENT="$2"
            shift 2
            ;;
        --segments-per-batch)
            SEGMENTS_PER_BATCH="$2"
            shift 2
            ;;
        --video-list-file)
            VIDEO_LIST_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --base-video-dir)
            BASE_VIDEO_DIR="$2"
            shift 2
            ;;
        --input-mask-dir)
            INPUT_MASK_DIR="$2"
            shift 2
            ;;
        --score-thresh)
            SCORE_THRESH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$VIDEO_LIST_FILE" ]] || [[ -z "$CHECKPOINT" ]] || [[ -z "$BASE_VIDEO_DIR" ]] || [[ -z "$INPUT_MASK_DIR" ]]; then
    echo "Usage: $0 --video-list-file <file> --checkpoint <path> --base-video-dir <dir> --input-mask-dir <dir> [--num-gpus <N>] [--num-frames-per-segment <N>] [--segments-per-batch <N>] [--output-dir <dir>] [--config <file>] [--score-thresh <value>]"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve relative paths (so the script can be run from anywhere)
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi
if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="$PROJECT_ROOT/$CHECKPOINT"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi

# Auto-select checkpoint if a directory is provided or if the file doesn't exist
if [[ -d "$CHECKPOINT" ]]; then
    CKPT_DIR="$CHECKPOINT"
    # prefer checkpoint.pt if present
    if [[ -f "$CKPT_DIR/checkpoint.pt" ]]; then
        CHECKPOINT="$CKPT_DIR/checkpoint.pt"
    else
        CHECKPOINT="$(ls -1t "$CKPT_DIR"/*.pt 2>/dev/null | head -n 1)"
    fi
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT"
    echo "Tip: pass --checkpoint <path/to/checkpoints_dir> or --checkpoint <path/to/checkpoint.pt>"
    exit 1
fi

# Create temporary directory for split video lists
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo "Splitting video list across $NUM_GPUS GPUs..."

# Read all video names line by line.
# IMPORTANT: avoid command-substitution like `VIDEOS=($(...))` because for large lists it can
# silently truncate (and it also splits on whitespace). Use `mapfile` with a process substitution.
# This keeps one video name per line and scales to large files.
mapfile -t VIDEOS < <(grep -v -E '^[[:space:]]*$' "$VIDEO_LIST_FILE")

# Split videos across GPUs
TOTAL_VIDEOS=${#VIDEOS[@]}
VIDEOS_PER_GPU=$(( (TOTAL_VIDEOS + NUM_GPUS - 1) / NUM_GPUS ))

echo "Total videos to process: $TOTAL_VIDEOS"
echo "Videos per GPU: ~$VIDEOS_PER_GPU"

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    START_IDX=$((gpu * VIDEOS_PER_GPU))
    END_IDX=$((START_IDX + VIDEOS_PER_GPU))
    
    if [[ $START_IDX -lt $TOTAL_VIDEOS ]]; then
        if [[ $END_IDX -gt $TOTAL_VIDEOS ]]; then
            END_IDX=$TOTAL_VIDEOS
        fi
        
        GPU_VIDEO_LIST="$TMP_DIR/videos_gpu${gpu}.txt"
        for ((i=START_IDX; i<END_IDX; i++)); do
            echo "${VIDEOS[$i]}" >> "$GPU_VIDEO_LIST"
        done
        
        ACTUAL_COUNT=$((END_IDX - START_IDX))
        echo "GPU $gpu: Processing videos $START_IDX to $((END_IDX-1)) ($ACTUAL_COUNT videos)"
    fi
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Launch processes for each GPU
echo "Launching processes on $NUM_GPUS GPUs..."
PIDS=()

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    GPU_VIDEO_LIST="$TMP_DIR/videos_gpu${gpu}.txt"
    
    if [[ -f "$GPU_VIDEO_LIST" ]]; then
        LOG_FILE="${OUTPUT_DIR}/test_inference_gpu${gpu}.log"
        
        (
            cd "$PROJECT_ROOT"
            export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
            export PYTHONUNBUFFERED=1
            CUDA_VISIBLE_DEVICES=$gpu python -u tools/test_video_segmentation.py \
                --sam2_cfg "$CONFIG_FILE" \
                --sam2_checkpoint "$CHECKPOINT" \
                --base_video_dir "$BASE_VIDEO_DIR" \
                --input_mask_dir "$INPUT_MASK_DIR" \
                --video_list_file "$GPU_VIDEO_LIST" \
                --output_mask_dir "$OUTPUT_DIR" \
                --segments_per_batch "$SEGMENTS_PER_BATCH" \
                --num_frames_per_segment "$NUM_FRAMES_PER_SEGMENT" \
                --score_thresh "$SCORE_THRESH" \
                > "$LOG_FILE" 2>&1
        ) &
        
        PIDS+=($!)
        echo "Started GPU $gpu process (PID: ${PIDS[$gpu]})"
    fi
done

echo ""
echo "All processes started. Monitoring progress..."
echo "Log files: ${OUTPUT_DIR}/test_inference_gpu*.log"
echo ""

# Wait for all processes to complete
for ((gpu=0; gpu<${#PIDS[@]}; gpu++)); do
    if [[ -n "${PIDS[$gpu]}" ]]; then
        wait ${PIDS[$gpu]}
        echo "GPU $gpu process completed (PID: ${PIDS[$gpu]})"
    fi
done

echo ""
echo "All processes completed!"
echo "Results saved to: $OUTPUT_DIR"

