#!/usr/bin/env python3
"""
CLI script to calculate metrics for mask-based video segmentation evaluation.

Usage:
    python calculate_mask_metrics.py --gt_folder <ground_truth_folder> --pred_folder <prediction_folder> [options]

Example:
    python calculate_mask_metrics.py \
        --gt_folder ./data/VQS-4K/masks \
        --pred_folder ./outputs/predictions \
        --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the metrics directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mask_metrics import compute_mask_based_metrics
from mask_loader import load_video_masks


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics for mask-based video segmentation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python calculate_mask_metrics.py \\
    --gt_folder /path/to/ground_truth/masks \\
    --pred_folder /path/to/prediction/masks

  # Save results to file
  python calculate_mask_metrics.py \\
    --gt_folder /path/to/ground_truth/masks \\
    --pred_folder /path/to/prediction/masks \\
    --output results.json

  # Verbose output
  python calculate_mask_metrics.py \\
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
    
    parser.add_argument(
        "--sample_video",
        type=str,
        default=None,
        help="Process only a specific video (for testing). Specify video folder name."
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
        metrics = compute_mask_based_metrics(args.gt_folder, args.pred_folder)
        
        # Prepare results
        results = {
            "ground_truth_folder": args.gt_folder,
            "prediction_folder": args.pred_folder,
            "metrics": metrics
        }
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            for metric_name, value in metrics.items():
                print(f"{metric_name:<40}: {value:.4f}")
            print("="*60)
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Total videos processed: {len(metrics) // 4}")  # Assuming 4 metrics per video
        print(f"- Metrics calculated: {len(metrics)}")
        
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


def test_single_video(gt_folder: str, pred_folder: str, video_name: str):
    """Test function to process a single video."""
    print(f"Testing with video: {video_name}")
    
    gt_path = os.path.join(gt_folder, video_name)
    pred_path = os.path.join(pred_folder, video_name)
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth folder for {video_name} does not exist: {gt_path}")
        return
        
    if not os.path.exists(pred_path):
        print(f"Error: Prediction folder for {video_name} does not exist: {pred_path}")
        return
    
    # Load masks for this video
    gt_masks = load_video_masks(gt_path, exclude_query_frames=True)
    pred_masks = load_video_masks(pred_path, exclude_query_frames=True)
    
    print(f"Ground truth frames: {len(gt_masks)}")
    print(f"Prediction frames: {len(pred_masks)}")
    
    if gt_masks:
        print(f"GT frame range: {min(gt_masks.keys())} - {max(gt_masks.keys())}")
    if pred_masks:
        print(f"Pred frame range: {min(pred_masks.keys())} - {max(pred_masks.keys())}")


if __name__ == "__main__":
    main()
