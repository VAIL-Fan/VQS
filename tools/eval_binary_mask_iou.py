#!/usr/bin/env python3
"""Compute per-frame binary IoU (Jaccard) between prediction and GT mask folders.

Predictions are expected as VIDEO_NAME/#####.png (e.g. from test_video_segmentation.py).
GT masks use the same layout; any non-zero GT label is treated as foreground.

Missing prediction PNGs are treated as an empty mask (all background).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image


def _class_from_video_name(video_name: str) -> str:
    m = re.search(r"-set_\d+$", video_name)
    if m:
        return video_name[: m.start()]
    return "unknown"


def _iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pb = pred > 0
    gb = gt > 0
    inter = np.logical_and(pb, gb).sum(dtype=np.int64)
    union = np.logical_or(pb, gb).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-list-file", type=str, required=True)
    p.add_argument("--pred-dir", type=str, required=True, help="Root dir of predicted masks")
    p.add_argument("--gt-dir", type=str, required=True, help="Root dir of GT masks")
    p.add_argument("--out-json", type=str, default="", help="Optional path to write JSON summary")
    args = p.parse_args()

    with open(args.video_list_file) as f:
        videos = [ln.strip() for ln in f if ln.strip()]

    per_video_mean: dict[str, float] = {}
    class_to_ious: dict[str, list[float]] = defaultdict(list)
    all_frame_ious: list[float] = []

    missing_pred_videos = 0
    for video in videos:
        gdir = os.path.join(args.gt_dir, video)
        pdir = os.path.join(args.pred_dir, video)
        if not os.path.isdir(gdir):
            print(f"[skip] missing GT dir: {gdir}")
            continue
        if not os.path.isdir(pdir):
            missing_pred_videos += 1
        pngs = sorted(
            x for x in os.listdir(gdir) if x.lower().endswith(".png")
        )
        if not pngs:
            print(f"[skip] no GT pngs: {gdir}")
            continue
        ious: list[float] = []
        for png in pngs:
            gt_path = os.path.join(gdir, png)
            pr_path = os.path.join(pdir, png)
            gt = np.array(Image.open(gt_path))
            if gt.ndim > 2:
                gt = gt[..., 0]
            if os.path.isfile(pr_path):
                pred = np.array(Image.open(pr_path))
                if pred.ndim > 2:
                    pred = pred[..., 0]
            else:
                pred = np.zeros_like(gt, dtype=np.uint8)
            if pred.shape != gt.shape:
                pred = np.array(
                    Image.fromarray(pred).resize(
                        (gt.shape[1], gt.shape[0]), Image.NEAREST
                    )
                )
            iou = _iou_binary(pred, gt)
            ious.append(iou)
            cls = _class_from_video_name(video)
            class_to_ious[cls].append(iou)
            all_frame_ious.append(iou)
        per_video_mean[video] = float(np.mean(ious)) if ious else float("nan")

    overall = float(np.mean(all_frame_ious)) if all_frame_ious else 0.0
    per_class = {
        cls: float(np.mean(vs)) for cls, vs in sorted(class_to_ious.items())
    }

    summary = {
        "num_videos_listed": len(videos),
        "num_videos_evaluated": len(per_video_mean),
        "missing_pred_video_dirs": missing_pred_videos,
        "num_frames": len(all_frame_ious),
        "mean_jaccard_iou": overall,
        "mean_jaccard_per_class": per_class,
        "mean_jaccard_per_video": per_video_mean,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "mean_jaccard_per_video"}, indent=2))
    print(f"\nMean Jaccard (IoU) over {len(all_frame_ious)} frames: {overall:.6f}")
    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as jf:
            json.dump(summary, jf, indent=2)
        print(f"Wrote full summary (incl. per-video) to {args.out_json}")


if __name__ == "__main__":
    main()
