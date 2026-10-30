# VQS Metric Definitions

This folder implements the official VQS evaluation metrics used in the project run scripts (`run_CVPR_24_putting.sh`, `run_ICCV_25_latest.sh`, etc.).

## Reported metrics

| Paper / log name | Script output name | Description |
|---|---|---|
| **tAP** | `Temporal AP @ IoU=0.25` | Temporal average precision (single top-1 prediction per video) |
| **tAP50** | `Temporal AP @ IoU=0.50` | tAP at IoU threshold 0.50 |
| **tAP75** | `Temporal AP @ IoU=0.75` | tAP at IoU threshold 0.75 |
| **stAP** | `SpatioTemporal AP @ IoU=0.25` | Spatio-temporal average precision |
| **stAP50** | `SpatioTemporal AP @ IoU=0.50` | stAP at IoU threshold 0.50 |
| **stAP75** | `SpatioTemporal AP @ IoU=0.75` | stAP at IoU threshold 0.75 |
| **Succ** | `Success @ IoU=0.05/0.10/0.20` | % of videos with st-IoU above threshold |
| **Rec** | `Tracking % recovery @ IoU=0.50/0.75/0.95` | % of GT frames with spatial IoU above threshold |
| **t-IoU** | `t-IoU` | Mean temporal IoU across videos |
| **st-IoU** | `st-IoU` | Mean spatio-temporal IoU across videos |

Thresholds:
- tAP / stAP: 0.25, 0.50, 0.75, 0.95
- Success: 0.05, 0.10, 0.20
- Recovery: 0.50, 0.75, 0.95

## Code layout

```
metrics/
├── calculate_mask_metrics_simple.py   # Main CLI used by all run_*.sh scripts
├── calculate_mask_metrics.py          # Alternative CLI using metrics/metrics library
├── mask_loader.py                     # Load PNG masks, convert to response tracks
├── mask_metrics.py                    # Mask wrappers around VQ2D metric classes
├── evaluation/structures.py           # BBox / ResponseTrack data structures
└── metrics/                           # Core VQ2D metric implementations
    ├── temporal_metrics.py            # tAP
    ├── spatio_temporal_metrics.py     # stAP
    ├── success_metrics.py             # Success
    └── tracking_metrics.py            # Recovery (% recovery)
```

## Quick start

```bash
cd metrics
pip install -r ../requirements.txt

python calculate_mask_metrics_simple.py \
  --gt_folder /path/to/vqs_4k_huggingface/masks \
  --pred_folder /path/to/model/predicted_masks \
  --output results.json \
  --verbose
```

Or with a test split filter:

```bash
bash scripts/run_metrics.sh \
  --gt-folder /path/to/vqs_4k_huggingface/masks \
  --pred-folder /path/to/model/predicted_masks \
  --video-list /path/to/test_t-IoU.txt \
  --output results.json
```

## Input format

Each video is a subfolder containing frame-wise binary mask PNG files:

```
masks/
├── human-basketball_player-set_2/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── ...
```

- Frame files are named `XXXXX.png` (zero-padded frame index).
- Query frames `00000.png` and `99999.png` are excluded automatically.
- Missing frames mean the object is absent in that frame.

## Notes

- With one prediction per video (top-1), tAP/stAP reduce to a pass/fail rate at each IoU threshold.
- AP values in `calculate_mask_metrics_simple.py` are returned as fractions in `[0, 1]`; Success and Recovery are percentages in `[0, 100]`.
- The lower-level `metrics/metrics/` package follows the original VQ2D bbox evaluation logic; mask evaluation uses the simplified top-1 implementation in `calculate_mask_metrics_simple.py`, which is what all project `run_*.sh` scripts call.
- **Recovery denominator:** `Tracking % recovery` is computed over **every ground-truth frame**. A GT frame for which the method produced no prediction counts as a miss, not as an excluded sample.
- **Query frames:** `00000.png` and `99999.png` are excluded from all metrics.

## Example result (official test split, 822 videos)

```
Temporal AP @ IoU=0.50                  : 0.2676
SpatioTemporal AP @ IoU=0.50            : 0.2384
Success @ IoU=0.05                      : 55.9611
Tracking % recovery @ IoU=0.50          : 43.5910
t-IoU                                   : 0.2960
st-IoU                                  : 0.2599
```

## Mapping to the names used in the paper

The paper's tables use `mSTIoU` / `mTIoU` naming. The correspondence is:

| Paper | Script output key | Scale |
|---|---|---|
| mSTIoU | `st-IoU` | x100 |
| mSTIoU50 | `SpatioTemporal AP @ IoU=0.50` | x100 |
| mSTIoU75 | `SpatioTemporal AP @ IoU=0.75` | x100 |
| mTIoU | `t-IoU` | x100 |
| mTIoU50 | `Temporal AP @ IoU=0.50` | x100 |
| mTIoU75 | `Temporal AP @ IoU=0.75` | x100 |
| Rec | `Tracking % recovery @ IoU=0.50` | as printed |
| Succ | `Success @ IoU=0.20` | as printed |

`Rec` is the percentage of ground-truth frames whose mask IoU exceeds 0.5; `Succ` is measured
at an IoU of 0.2. Neither aggregates across thresholds -- the other thresholds the script
prints are reported only in the appendix.

Running the above on the released VQ-SAM predictions and the released ground truth over
`splits/test_t-IoU.txt` reproduces every column of the paper's main table:

| | mSTIoU | mSTIoU50 | mSTIoU75 | mTIoU | mTIoU50 | mTIoU75 | Rec | Succ |
|---|---|---|---|---|---|---|---|---|
| Paper | 26.0 | 23.8 | 12.7 | 29.6 | 26.8 | 14.4 | 43.6 | 42.1 |
| This code | 25.99 | 23.84 | 12.65 | 29.60 | 26.76 | 14.36 | 43.59 | 42.09 |
