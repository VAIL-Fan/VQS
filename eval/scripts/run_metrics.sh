#!/bin/bash
# Evaluate predicted masks against VQS ground-truth masks.
#
# Usage:
#   ./run_metrics.sh \
#     --gt-folder /path/to/gt/masks \
#     --pred-folder /path/to/pred/masks \
#     [--output results.json] \
#     [--video-list /path/to/test.txt]
#
# If --video-list is provided, only videos listed in that file are evaluated
# (symlinks are created in a temporary directory).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GT_FOLDER=""
PRED_FOLDER=""
OUTPUT_FILE=""
VIDEO_LIST_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gt-folder)
            GT_FOLDER="$2"
            shift 2
            ;;
        --pred-folder)
            PRED_FOLDER="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --video-list)
            VIDEO_LIST_FILE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$GT_FOLDER" || -z "$PRED_FOLDER" ]]; then
    echo "Error: --gt-folder and --pred-folder are required."
    exit 1
fi

if [[ ! -d "$GT_FOLDER" ]]; then
    echo "Error: GT folder not found: $GT_FOLDER"
    exit 1
fi

if [[ ! -d "$PRED_FOLDER" ]]; then
    echo "Error: prediction folder not found: $PRED_FOLDER"
    exit 1
fi

# Resolve to absolute paths. Below we may symlink videos into a temporary directory
# and we cd into the metrics root before invoking python, so relative paths given on
# the command line would otherwise dangle.
GT_FOLDER="$(cd "$GT_FOLDER" && pwd)"
PRED_FOLDER="$(cd "$PRED_FOLDER" && pwd)"
if [[ -n "$VIDEO_LIST_FILE" && -f "$VIDEO_LIST_FILE" ]]; then
    VIDEO_LIST_FILE="$(cd "$(dirname "$VIDEO_LIST_FILE")" && pwd)/$(basename "$VIDEO_LIST_FILE")"
fi
if [[ -n "$OUTPUT_FILE" ]]; then
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    OUTPUT_FILE="$(cd "$(dirname "$OUTPUT_FILE")" && pwd)/$(basename "$OUTPUT_FILE")"
fi

EVAL_GT="$GT_FOLDER"
EVAL_PRED="$PRED_FOLDER"
TEMP_DIR=""

if [[ -n "$VIDEO_LIST_FILE" ]]; then
    if [[ ! -f "$VIDEO_LIST_FILE" ]]; then
        echo "Error: video list not found: $VIDEO_LIST_FILE"
        exit 1
    fi

    TEMP_DIR="$(mktemp -d)"
    EVAL_GT="$TEMP_DIR/gt_masks"
    EVAL_PRED="$TEMP_DIR/pred_masks"
    mkdir -p "$EVAL_GT" "$EVAL_PRED"

    echo "Filtering videos using: $VIDEO_LIST_FILE"
    video_count=0
    while IFS= read -r video_name || [[ -n "$video_name" ]]; do
        video_name="${video_name//$'\r'/}"
        [[ -z "$video_name" ]] && continue

        if [[ -d "$GT_FOLDER/$video_name" && -d "$PRED_FOLDER/$video_name" ]]; then
            ln -s "$GT_FOLDER/$video_name" "$EVAL_GT/$video_name"
            ln -s "$PRED_FOLDER/$video_name" "$EVAL_PRED/$video_name"
            video_count=$((video_count + 1))
        else
            echo "  Warning: skipping $video_name (missing in GT or prediction folder)"
        fi
    done < "$VIDEO_LIST_FILE"

    echo "Filtered $video_count videos for evaluation"
fi

echo "Ground truth: $EVAL_GT"
echo "Predictions:  $EVAL_PRED"
if [[ -n "$OUTPUT_FILE" ]]; then
    echo "Output:       $OUTPUT_FILE"
fi
echo ""

cd "$METRICS_ROOT"
python calculate_mask_metrics_simple.py \
    --gt_folder "$EVAL_GT" \
    --pred_folder "$EVAL_PRED" \
    ${OUTPUT_FILE:+--output "$OUTPUT_FILE"} \
    --verbose

if [[ -n "$TEMP_DIR" ]]; then
    rm -rf "$TEMP_DIR"
fi

echo ""
echo "Metrics calculation completed."
