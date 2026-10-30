#!/usr/bin/env python3
"""
Simplified CLI script to calculate metrics for mask-based video segmentation evaluation.

This script calculates the 4 main metrics:
1. Temporal AP - temporal detection performance
2. SpatioTemporal AP - combined spatial and temporal performance  
3. Success Metrics - success rate with different IoU thresholds
4. Tracking Metrics - frame-level tracking recovery

Usage:
    python calculate_mask_metrics_simple.py --gt_folder <ground_truth_folder> --pred_folder <prediction_folder> [options]
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from collections import OrderedDict


def load_mask_from_path(mask_path: str) -> np.ndarray:
    """Load a binary mask from PNG file."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    return mask


def load_video_masks(mask_folder: str, exclude_query_frames: bool = True) -> Dict[int, np.ndarray]:
    """
    Load all masks for a video from a folder.
    
    Args:
        mask_folder: Path to folder containing mask PNG files
        exclude_query_frames: If True, exclude 00000.png and 99999.png (query frames)
    
    Returns:
        Dictionary mapping frame numbers to mask arrays
    """
    mask_files = glob.glob(os.path.join(mask_folder, "*.png"))
    masks = {}
    
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        frame_num = int(filename.split('.')[0])
        
        # Skip query frames if requested
        if exclude_query_frames and frame_num in [0, 99999]:
            continue
            
        try:
            mask = load_mask_from_path(mask_file)
            masks[frame_num] = mask
        except Exception as e:
            print(f"Warning: Could not load {mask_file}: {e}")
            continue
    
    return masks


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate IoU between two binary masks."""
    # Ensure masks are binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_temporal_iou(gt_frames: set, pred_frames: set) -> float:
    """Calculate temporal IoU between ground truth and prediction frames."""
    intersection_frames = len(gt_frames & pred_frames)
    union_frames = len(gt_frames | pred_frames)
    
    if union_frames == 0:
        return 0.0
    
    return intersection_frames / union_frames


def calculate_tiou_metric(gt_masks_list: List[Dict[int, np.ndarray]], 
                          pred_masks_list: List[Dict[int, np.ndarray]]) -> float:
    """
    Calculate t-IoU (temporal IoU) metric.
    
    For each video: t-IoU = intersection_frames / union_frames
    Final metric: average t-IoU across all videos
    
    Args:
        gt_masks_list: List of ground truth mask dictionaries per video
        pred_masks_list: List of prediction mask dictionaries per video
    
    Returns:
        Average t-IoU across all videos
    """
    tiou_scores = []
    
    for gt_masks, pred_masks in zip(gt_masks_list, pred_masks_list):
        if not pred_masks:  # No prediction
            tiou_scores.append(0.0)
            continue
        
        # Get frame sets
        gt_frames = set(gt_masks.keys())
        pred_frames = set(pred_masks.keys())
        
        # Calculate t-IoU for this video
        tiou = calculate_temporal_iou(gt_frames, pred_frames)
        tiou_scores.append(tiou)
    
    # Return average t-IoU across all videos
    return np.mean(tiou_scores) if tiou_scores else 0.0


def calculate_stiou_metric(gt_masks_list: List[Dict[int, np.ndarray]], 
                           pred_masks_list: List[Dict[int, np.ndarray]]) -> float:
    """
    Calculate st-IoU (spatio-temporal IoU) metric using pixel-based calculation.
    
    For each video: st-IoU = area_intersection / (area_gt + area_pred - area_intersection)
    where:
    - area_intersection: sum of intersection pixels across intersection frames only
    - area_gt: total pixels in ground truth across ALL GT frames
    - area_pred: total pixels in prediction across ALL prediction frames
    
    Args:
        gt_masks_list: List of ground truth mask dictionaries per video
        pred_masks_list: List of prediction mask dictionaries per video
    
    Returns:
        Average st-IoU across all videos
    """
    stiou_scores = []
    
    for gt_masks, pred_masks in zip(gt_masks_list, pred_masks_list):
        if not pred_masks:  # No prediction
            stiou_scores.append(0.0)
            continue
        
        # Find intersection frames (frames that exist in both GT and prediction)
        gt_frames = set(gt_masks.keys())
        pred_frames = set(pred_masks.keys())
        intersection_frames = gt_frames & pred_frames
        
        if not intersection_frames:
            stiou_scores.append(0.0)
            continue
        
        # Calculate intersection area (only in intersection frames)
        total_intersection = 0
        for frame_num in intersection_frames:
            gt_mask = gt_masks[frame_num]
            pred_mask = pred_masks[frame_num]
            
            # Ensure masks have same shape
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
            
            # Convert to binary
            gt_binary = (gt_mask > 0).astype(np.uint8)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            
            # Calculate intersection for this frame
            intersection = np.logical_and(gt_binary, pred_binary).sum()
            total_intersection += intersection
        
        # Calculate total GT area (across ALL GT frames)
        total_gt_area = 0
        for frame_num in gt_frames:
            gt_mask = gt_masks[frame_num]
            gt_binary = (gt_mask > 0).astype(np.uint8)
            total_gt_area += gt_binary.sum()
        
        # Calculate total prediction area (across ALL prediction frames)
        total_pred_area = 0
        for frame_num in pred_frames:
            pred_mask = pred_masks[frame_num]
            pred_binary = (pred_mask > 0).astype(np.uint8)
            total_pred_area += pred_binary.sum()
        
        # Calculate st-IoU for this video
        if total_gt_area + total_pred_area - total_intersection == 0:
            stiou_scores.append(0.0)
        else:
            stiou = total_intersection / (total_gt_area + total_pred_area - total_intersection)
            stiou_scores.append(stiou)
    
    # Return average st-IoU across all videos
    return np.mean(stiou_scores) if stiou_scores else 0.0


def calculate_spatio_temporal_iou(gt_masks: Dict[int, np.ndarray], 
                                pred_masks: Dict[int, np.ndarray]) -> float:
    """
    Calculate spatio-temporal IoU between ground truth and prediction masks.
    
    Corrected calculation:
    - area_intersection: sum of intersection pixels across intersection frames only
    - area_gt: total pixels in ground truth across ALL GT frames
    - area_pred: total pixels in prediction across ALL prediction frames
    - st-IoU = area_intersection / (area_gt + area_pred - area_intersection)
    """
    # Find intersection frames (frames that exist in both GT and prediction)
    gt_frames = set(gt_masks.keys())
    pred_frames = set(pred_masks.keys())
    intersection_frames = gt_frames & pred_frames
    
    if not intersection_frames:
        return 0.0
    
    # Calculate intersection area (only in intersection frames)
    total_intersection = 0
    for frame_num in intersection_frames:
        gt_mask = gt_masks[frame_num]
        pred_mask = pred_masks[frame_num]
        
        # Ensure masks have same shape
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # Convert to binary
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)
        
        # Calculate intersection for this frame
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        total_intersection += intersection
    
    # Calculate total GT area (across ALL GT frames)
    total_gt_area = 0
    for frame_num in gt_frames:
        gt_mask = gt_masks[frame_num]
        gt_binary = (gt_mask > 0).astype(np.uint8)
        total_gt_area += gt_binary.sum()
    
    # Calculate total prediction area (across ALL prediction frames)
    total_pred_area = 0
    for frame_num in pred_frames:
        pred_mask = pred_masks[frame_num]
        pred_binary = (pred_mask > 0).astype(np.uint8)
        total_pred_area += pred_binary.sum()
    
    # Calculate st-IoU
    if total_gt_area + total_pred_area - total_intersection == 0:
        return 0.0
    
    return total_intersection / (total_gt_area + total_pred_area - total_intersection)


def calculate_temporal_ap(gt_masks_list: List[Dict[int, np.ndarray]], 
                         pred_masks_list: List[Dict[int, np.ndarray]],
                         thresholds: List[float] = [0.25, 0.5, 0.75, 0.95]) -> Dict[str, float]:
    """
    Calculate Temporal AP metrics for multi-clip segmentation.
    
    For each video, calculates temporal IoU and determines if it meets the threshold.
    Uses frame-based temporal IoU calculation suitable for discontinuous clips.
    """
    results = {}
    
    for threshold in thresholds:
        tp_count = 0
        fp_count = 0
        
        for gt_masks, pred_masks in zip(gt_masks_list, pred_masks_list):
            if not pred_masks:  # No prediction
                fp_count += 1
                continue
            
            # Calculate temporal IoU for multi-clip segmentation
            gt_frames = set(gt_masks.keys())
            pred_frames = set(pred_masks.keys())
            tiou = calculate_temporal_iou(gt_frames, pred_frames)
            
            if tiou >= threshold:
                tp_count += 1
            else:
                fp_count += 1
        
        # Calculate precision
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        results[f"Temporal AP @ IoU={threshold:.2f}"] = precision
    
    # Average AP
    avg_ap = np.mean(list(results.values()))
    results["Temporal AP @ IoU=0.25:0.95"] = avg_ap
    
    return results


def calculate_spatio_temporal_ap(gt_masks_list: List[Dict[int, np.ndarray]], 
                                pred_masks_list: List[Dict[int, np.ndarray]],
                                thresholds: List[float] = [0.25, 0.5, 0.75, 0.95]) -> Dict[str, float]:
    """
    Calculate SpatioTemporal AP metrics for multi-clip segmentation using pixel-based IoU.
    
    For each video, calculates spatio-temporal IoU using pixel-based intersection/union
    across all frames, suitable for discontinuous clips and mask-based evaluation.
    """
    results = {}
    
    for threshold in thresholds:
        tp_count = 0
        fp_count = 0
        
        for gt_masks, pred_masks in zip(gt_masks_list, pred_masks_list):
            if not pred_masks:  # No prediction
                fp_count += 1
                continue
            
            # Calculate pixel-based spatio-temporal IoU for multi-clip segmentation
            stiou = calculate_spatio_temporal_iou(gt_masks, pred_masks)
            
            if stiou >= threshold:
                tp_count += 1
            else:
                fp_count += 1
        
        # Calculate precision
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        results[f"SpatioTemporal AP @ IoU={threshold:.2f}"] = precision
    
    # Average AP
    avg_ap = np.mean(list(results.values()))
    results["SpatioTemporal AP @ IoU=0.25:0.95"] = avg_ap
    
    return results


def calculate_success_metrics(gt_masks_list: List[Dict[int, np.ndarray]], 
                             pred_masks_list: List[Dict[int, np.ndarray]],
                             thresholds: List[float] = [0.05, 0.1, 0.2]) -> Dict[str, float]:
    """
    Calculate Success metrics for multi-clip segmentation using pixel-based IoU.
    
    Measures success rate based on pixel-based spatio-temporal IoU thresholds,
    suitable for discontinuous clips and mask-based evaluation.
    """
    results = {}
    
    for threshold in thresholds:
        success_count = 0
        total_count = len(gt_masks_list)
        
        for gt_masks, pred_masks in zip(gt_masks_list, pred_masks_list):
            if not pred_masks:  # No prediction
                continue
            
            # Calculate pixel-based spatio-temporal IoU for multi-clip segmentation
            stiou = calculate_spatio_temporal_iou(gt_masks, pred_masks)
            
            if stiou >= threshold:
                success_count += 1
        
        success_rate = (success_count / total_count) * 100.0 if total_count > 0 else 0.0
        results[f"Success @ IoU={threshold:.2f}"] = success_rate
    
    return results


def calculate_tracking_metrics(gt_masks_list: List[Dict[int, np.ndarray]], 
                              pred_masks_list: List[Dict[int, np.ndarray]],
                              thresholds: List[float] = [0.5, 0.75, 0.95]) -> Dict[str, float]:
    """
    Calculate Tracking recovery metrics for multi-clip segmentation using pixel-based IoU.
    
    Measures frame-level tracking recovery using pixel-based IoU per frame,
    suitable for discontinuous clips and mask-based evaluation.
    """
    results = {}
    
    for threshold in thresholds:
        total_accurate_frames = 0
        total_frames = 0
        
        for gt_masks, pred_masks in zip(gt_masks_list, pred_masks_list):
            if not pred_masks:  # No prediction
                total_frames += len(gt_masks)
                continue
            
            # Find common frames
            common_frames = set(gt_masks.keys()) & set(pred_masks.keys())
            
            for frame_num in common_frames:
                gt_mask = gt_masks[frame_num]
                pred_mask = pred_masks[frame_num]
                
                # Ensure masks have same shape
                if gt_mask.shape != pred_mask.shape:
                    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                
                # Calculate pixel-based IoU for this frame
                frame_iou = calculate_mask_iou(gt_mask, pred_mask)
                
                if frame_iou >= threshold:
                    total_accurate_frames += 1
                
                total_frames += 1
        
        recovery_rate = (total_accurate_frames / total_frames) * 100.0 if total_frames > 0 else 0.0
        results[f"Tracking % recovery @ IoU={threshold:.2f}"] = recovery_rate
    
    return results


def compute_all_metrics(gt_folder: str, pred_folder: str) -> Tuple[Dict[str, float], List[Dict[str, any]]]:
    """
    Compute all 6 metrics for mask-based evaluation:
    1. Temporal AP - temporal detection performance
    2. SpatioTemporal AP - combined spatial and temporal performance  
    3. Success Metrics - success rate with different IoU thresholds
    4. Tracking Metrics - frame-level tracking recovery
    5. t-IoU - average temporal IoU across all videos
    6. st-IoU - average spatio-temporal IoU across all videos
    
    Args:
        gt_folder: Path to ground truth masks folder
        pred_folder: Path to prediction masks folder
    
    Returns:
        Tuple of (overall_metrics, per_video_results)
    """
    print("Loading mask data...")
    
    # Load all masks
    gt_masks_list = []
    pred_masks_list = []
    video_names = []
    
    # Get all video folders
    gt_videos = [d for d in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, d))]
    pred_videos = [d for d in os.listdir(pred_folder) if os.path.isdir(os.path.join(pred_folder, d))]
    
    # Find common videos
    common_videos = set(gt_videos) & set(pred_videos)
    print(f"Found {len(common_videos)} common videos")
    
    for video_id in sorted(common_videos):
        # Load ground truth masks
        gt_folder_path = os.path.join(gt_folder, video_id)
        gt_masks = load_video_masks(gt_folder_path, exclude_query_frames=True)
        
        # Load prediction masks
        pred_folder_path = os.path.join(pred_folder, video_id)
        pred_masks = load_video_masks(pred_folder_path, exclude_query_frames=True)
        
        if gt_masks:  # Only process if ground truth has masks
            gt_masks_list.append(gt_masks)
            pred_masks_list.append(pred_masks)
            video_names.append(video_id)
    
    print(f"Loaded {len(gt_masks_list)} videos for evaluation")
    
    # Calculate per-video metrics
    per_video_results = []
    for i, (video_name, gt_masks, pred_masks) in enumerate(zip(video_names, gt_masks_list, pred_masks_list)):
        print(f"Processing video {i+1}/{len(video_names)}: {video_name}")
        
        # Calculate metrics for this video
        video_metrics = {}
        
        # Temporal IoU for this video
        if pred_masks:
            gt_frames = set(gt_masks.keys())
            pred_frames = set(pred_masks.keys())
            video_metrics['t-IoU'] = calculate_temporal_iou(gt_frames, pred_frames)
            
            # Spatio-temporal IoU for this video (corrected calculation)
            gt_frames = set(gt_masks.keys())
            pred_frames = set(pred_masks.keys())
            intersection_frames = gt_frames & pred_frames
            
            if intersection_frames:
                # Calculate intersection area (only in intersection frames)
                total_intersection = 0
                for frame_num in intersection_frames:
                    gt_mask = gt_masks[frame_num]
                    pred_mask = pred_masks[frame_num]
                    
                    if gt_mask.shape != pred_mask.shape:
                        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                    
                    gt_binary = (gt_mask > 0).astype(np.uint8)
                    pred_binary = (pred_mask > 0).astype(np.uint8)
                    intersection = np.logical_and(gt_binary, pred_binary).sum()
                    total_intersection += intersection
                
                # Calculate total GT area (across ALL GT frames)
                total_gt_area = 0
                for frame_num in gt_frames:
                    gt_mask = gt_masks[frame_num]
                    gt_binary = (gt_mask > 0).astype(np.uint8)
                    total_gt_area += gt_binary.sum()
                
                # Calculate total prediction area (across ALL prediction frames)
                total_pred_area = 0
                for frame_num in pred_frames:
                    pred_mask = pred_masks[frame_num]
                    pred_binary = (pred_mask > 0).astype(np.uint8)
                    total_pred_area += pred_binary.sum()
                
                # Calculate st-IoU
                if total_gt_area + total_pred_area - total_intersection == 0:
                    video_metrics['st-IoU'] = 0.0
                else:
                    video_metrics['st-IoU'] = total_intersection / (total_gt_area + total_pred_area - total_intersection)
            else:
                video_metrics['st-IoU'] = 0.0
            
            # Success metrics for this video
            stiou = video_metrics['st-IoU']
            video_metrics['Success@0.05'] = 1.0 if stiou >= 0.05 else 0.0
            video_metrics['Success@0.10'] = 1.0 if stiou >= 0.10 else 0.0
            video_metrics['Success@0.20'] = 1.0 if stiou >= 0.20 else 0.0
            
            # Tracking metrics for this video
            common_frames = set(gt_masks.keys()) & set(pred_masks.keys())
            if common_frames:
                accurate_frames_50 = 0
                accurate_frames_75 = 0
                accurate_frames_95 = 0
                
                for frame_num in common_frames:
                    gt_mask = gt_masks[frame_num]
                    pred_mask = pred_masks[frame_num]
                    
                    if gt_mask.shape != pred_mask.shape:
                        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
                    
                    frame_iou = calculate_mask_iou(gt_mask, pred_mask)
                    
                    if frame_iou >= 0.5:
                        accurate_frames_50 += 1
                    if frame_iou >= 0.75:
                        accurate_frames_75 += 1
                    if frame_iou >= 0.95:
                        accurate_frames_95 += 1
                
                total_frames = len(common_frames)
                video_metrics['Tracking@0.50'] = (accurate_frames_50 / total_frames) * 100.0
                video_metrics['Tracking@0.75'] = (accurate_frames_75 / total_frames) * 100.0
                video_metrics['Tracking@0.95'] = (accurate_frames_95 / total_frames) * 100.0
            else:
                video_metrics['Tracking@0.50'] = 0.0
                video_metrics['Tracking@0.75'] = 0.0
                video_metrics['Tracking@0.95'] = 0.0
        else:
            # No prediction
            video_metrics['t-IoU'] = 0.0
            video_metrics['st-IoU'] = 0.0
            video_metrics['Success@0.05'] = 0.0
            video_metrics['Success@0.10'] = 0.0
            video_metrics['Success@0.20'] = 0.0
            video_metrics['Tracking@0.50'] = 0.0
            video_metrics['Tracking@0.75'] = 0.0
            video_metrics['Tracking@0.95'] = 0.0
        
        per_video_results.append({
            'video_name': video_name,
            'metrics': video_metrics
        })
    
    # Calculate overall metrics
    all_metrics = OrderedDict()
    
    print("Calculating Temporal AP...")
    temporal_metrics = calculate_temporal_ap(gt_masks_list, pred_masks_list)
    all_metrics.update(temporal_metrics)
    
    print("Calculating SpatioTemporal AP...")
    spatio_temporal_metrics = calculate_spatio_temporal_ap(gt_masks_list, pred_masks_list)
    all_metrics.update(spatio_temporal_metrics)
    
    print("Calculating Success Metrics...")
    success_metrics = calculate_success_metrics(gt_masks_list, pred_masks_list)
    all_metrics.update(success_metrics)
    
    print("Calculating Tracking Metrics...")
    tracking_metrics = calculate_tracking_metrics(gt_masks_list, pred_masks_list)
    all_metrics.update(tracking_metrics)
    
    print("Calculating t-IoU metric...")
    tiou_score = calculate_tiou_metric(gt_masks_list, pred_masks_list)
    all_metrics["t-IoU"] = tiou_score
    
    print("Calculating st-IoU metric...")
    stiou_score = calculate_stiou_metric(gt_masks_list, pred_masks_list)
    all_metrics["st-IoU"] = stiou_score
    
    return all_metrics, per_video_results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics for mask-based video segmentation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python calculate_mask_metrics_simple.py \\
    --gt_folder /path/to/ground_truth/masks \\
    --pred_folder /path/to/prediction/masks

  # Save results to file
  python calculate_mask_metrics_simple.py \\
    --gt_folder /path/to/ground_truth/masks \\
    --pred_folder /path/to/prediction/masks \\
    --output results.json

  # Verbose output
  python calculate_mask_metrics_simple.py \\
    --gt_folder /path/to/ground_truth/masks \\
    --pred_folder /path/to/prediction/masks \\
    --verbose
        """
    )
    
    parser.add_argument(
        "--gt_folder",
        type=str,
        required=True,
        help="Path to ground truth masks folder (each subfolder is a video)"
    )
    
    parser.add_argument(
        "--pred_folder", 
        type=str,
        required=True,
        help="Path to prediction masks folder (each subfolder is a video)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (JSON format). If not specified, results are printed to stdout."
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input folders
    if not os.path.exists(args.gt_folder):
        print(f"Error: Ground truth folder does not exist: {args.gt_folder}")
        sys.exit(1)
        
    if not os.path.exists(args.pred_folder):
        print(f"Error: Prediction folder does not exist: {args.pred_folder}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Ground truth folder: {args.gt_folder}")
        print(f"Prediction folder: {args.pred_folder}")
        print(f"Output file: {args.output}")
        print()
    
    try:
        # Calculate metrics
        print("Starting metric calculation...")
        metrics, per_video_results = compute_all_metrics(args.gt_folder, args.pred_folder)
        
        # Prepare results
        results = {
            "ground_truth_folder": args.gt_folder,
            "prediction_folder": args.pred_folder,
            "metrics": metrics,
            "per_video_results": per_video_results
        }
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
            
            # Also create a text file with per-video results
            txt_output = args.output.replace('.json', '_per_video.txt')
            with open(txt_output, 'w') as f:
                f.write("Video_Name\tt-IoU\tst-IoU\tSuccess@0.05\tSuccess@0.10\tSuccess@0.20\tTracking@0.50\tTracking@0.75\tTracking@0.95\n")
                
                for video_result in per_video_results:
                    video_name = video_result['video_name']
                    video_metrics = video_result['metrics']
                    
                    f.write(f"{video_name}\t")
                    f.write(f"{video_metrics['t-IoU']:.4f}\t")
                    f.write(f"{video_metrics['st-IoU']:.4f}\t")
                    f.write(f"{video_metrics['Success@0.05']:.4f}\t")
                    f.write(f"{video_metrics['Success@0.10']:.4f}\t")
                    f.write(f"{video_metrics['Success@0.20']:.4f}\t")
                    f.write(f"{video_metrics['Tracking@0.50']:.4f}\t")
                    f.write(f"{video_metrics['Tracking@0.75']:.4f}\t")
                    f.write(f"{video_metrics['Tracking@0.95']:.4f}\n")
            
            print(f"Per-video results saved to: {txt_output}")
        else:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            for metric_name, value in metrics.items():
                print(f"{metric_name:<40}: {value:.4f}")
            print("="*60)
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Total videos processed: {len(per_video_results)}")
        print(f"- Overall metrics calculated: {len(metrics)}")
        
        if args.verbose:
            print(f"\nDetailed metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.6f}")
                
    except Exception as e:
        print(f"Error during metric calculation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
