import os
import numpy as np
import cv2
from typing import List, Dict, Tuple
from collections import OrderedDict

try:
    from .mask_loader import load_all_video_masks, load_video_masks, masks_to_spatio_temporal_iou, calculate_mask_temporal_iou, calculate_mask_iou
    from .spatio_temporal_metrics import SpatioTemporalDetection
    from .success_metrics import SuccessMetrics
    from .temporal_metrics import TemporalDetection
    from .tracking_metrics import TrackingMetrics
    from .utils import BBox, ResponseTrack
except ImportError:
    try:
        from mask_loader import load_all_video_masks, load_video_masks, masks_to_spatio_temporal_iou, calculate_mask_temporal_iou, calculate_mask_iou
        from metrics.spatio_temporal_metrics import SpatioTemporalDetection
        from metrics.success_metrics import SuccessMetrics
        from metrics.temporal_metrics import TemporalDetection
        from metrics.tracking_metrics import TrackingMetrics
        from evaluation.structures import BBox, ResponseTrack
    except ImportError:
        from mask_loader import load_all_video_masks, load_video_masks, masks_to_spatio_temporal_iou, calculate_mask_temporal_iou, calculate_mask_iou
        from metrics.spatio_temporal_metrics import SpatioTemporalDetection
        from metrics.success_metrics import SuccessMetrics
        from metrics.temporal_metrics import TemporalDetection
        from metrics.tracking_metrics import TrackingMetrics
        from utils import BBox, ResponseTrack


class MaskBasedSpatioTemporalDetection(SpatioTemporalDetection):
    """SpatioTemporal Detection metrics adapted for mask-based evaluation."""
    
    def __init__(self, gt_masks_list: List[Dict[int, np.ndarray]], 
                 pred_masks_list: List[Dict[int, np.ndarray]]):
        super().__init__([], [])
        self.gt_masks_list = gt_masks_list
        self.pred_masks_list = pred_masks_list
        self.ap = None
        self.ignore_iou_averaging = False

    def evaluate(self) -> None:
        """Evaluate using mask-based spatio-temporal IoU."""
        if not self.pred_masks_list:
            self.ap = np.zeros(len(self.iou_thresholds))
            self.average_ap = 0.0
            return
            
        npos = len(self.gt_masks_list)
        ap_scores = []
        
        for iou_thr in self.iou_thresholds:
            tp_count = 0
            fp_count = 0
            
            for i, (gt_masks, pred_masks) in enumerate(zip(self.gt_masks_list, self.pred_masks_list)):
                if not pred_masks:  # No prediction
                    fp_count += 1
                    continue
                    
                # Calculate spatio-temporal IoU
                stiou = masks_to_spatio_temporal_iou(gt_masks, pred_masks)
                
                if stiou >= iou_thr:
                    tp_count += 1
                else:
                    fp_count += 1
            
            # Calculate precision and recall
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / npos if npos > 0 else 0
            
            # For simplicity, use precision as AP (can be improved with proper PR curve)
            ap_scores.append(precision)
        
        self.ap = np.array(ap_scores)
        self.average_ap = self.ap.mean().item()


class MaskBasedTemporalDetection(TemporalDetection):
    """Temporal Detection metrics adapted for mask-based evaluation."""
    
    def __init__(self, gt_masks_list: List[Dict[int, np.ndarray]], 
                 pred_masks_list: List[Dict[int, np.ndarray]]):
        super().__init__([], [])
        self.gt_masks_list = gt_masks_list
        self.pred_masks_list = pred_masks_list
        self.ap = None
        self.ignore_iou_averaging = False

    def evaluate(self) -> None:
        """Evaluate using mask-based temporal IoU."""
        if not self.pred_masks_list:
            self.ap = np.zeros(len(self.tiou_thresholds))
            self.average_ap = 0.0
            return
            
        npos = len(self.gt_masks_list)
        ap_scores = []
        
        for tiou_thr in self.tiou_thresholds:
            tp_count = 0
            fp_count = 0
            
            for i, (gt_masks, pred_masks) in enumerate(zip(self.gt_masks_list, self.pred_masks_list)):
                if not pred_masks:  # No prediction
                    fp_count += 1
                    continue
                    
                # Calculate temporal IoU
                tiou = calculate_mask_temporal_iou(gt_masks, pred_masks)
                
                if tiou >= tiou_thr:
                    tp_count += 1
                else:
                    fp_count += 1
            
            # Calculate precision and recall
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / npos if npos > 0 else 0
            
            # For simplicity, use precision as AP
            ap_scores.append(precision)
        
        self.ap = np.array(ap_scores)
        self.average_ap = self.ap.mean().item()


class MaskBasedSuccessMetrics(SuccessMetrics):
    """Success metrics adapted for mask-based evaluation."""
    
    def __init__(self, gt_masks_list: List[Dict[int, np.ndarray]], 
                 pred_masks_list: List[Dict[int, np.ndarray]]):
        super().__init__([], [])
        self.gt_masks_list = gt_masks_list
        self.pred_masks_list = pred_masks_list
        self.success = None
        self.ignore_iou_averaging = False

    def evaluate(self) -> None:
        """Evaluate success rate using mask-based spatio-temporal IoU."""
        if not self.pred_masks_list:
            self.success = np.zeros(len(self.iou_thresholds))
            self.average_success = 0.0
            return
            
        success_scores = []
        
        for iou_thr in self.iou_thresholds:
            success_count = 0
            total_count = len(self.gt_masks_list)
            
            for gt_masks, pred_masks in zip(self.gt_masks_list, self.pred_masks_list):
                if not pred_masks:  # No prediction
                    continue
                    
                # Calculate spatio-temporal IoU
                stiou = masks_to_spatio_temporal_iou(gt_masks, pred_masks)
                
                if stiou >= iou_thr:
                    success_count += 1
            
            success_rate = (success_count / total_count) * 100.0 if total_count > 0 else 0.0
            success_scores.append(success_rate)
        
        self.success = np.array(success_scores)
        self.average_success = self.success.mean().item()


class MaskBasedTrackingMetrics(TrackingMetrics):
    """Tracking metrics adapted for mask-based evaluation."""
    
    def __init__(self, gt_masks_list: List[Dict[int, np.ndarray]], 
                 pred_masks_list: List[Dict[int, np.ndarray]]):
        super().__init__([], [])
        self.gt_masks_list = gt_masks_list
        self.pred_masks_list = pred_masks_list
        self.tracking_metrics = None
        self.ignore_iou_averaging = False

    def evaluate(self) -> None:
        """Evaluate tracking recovery using mask-based IoU."""
        if not self.pred_masks_list:
            self.tracking_metrics = {"% recovery": np.zeros(len(self.iou_thresholds))}
            self.average_tracking_metrics = {"% recovery": 0.0}
            return
            
        recovery_scores = []
        
        for iou_thr in self.iou_thresholds:
            total_accurate_frames = 0
            total_frames = 0
            
            for gt_masks, pred_masks in zip(self.gt_masks_list, self.pred_masks_list):
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
                    
                    # Calculate IoU for this frame
                    frame_iou = calculate_mask_iou(gt_mask, pred_mask)
                    
                    if frame_iou >= iou_thr:
                        total_accurate_frames += 1
                    
                    total_frames += 1
            
            recovery_rate = (total_accurate_frames / total_frames) * 100.0 if total_frames > 0 else 0.0
            recovery_scores.append(recovery_rate)
        
        self.tracking_metrics = {"% recovery": np.array(recovery_scores)}
        self.average_tracking_metrics = {"% recovery": np.array(recovery_scores).mean().item()}


def compute_mask_based_metrics(gt_folder: str, pred_folder: str) -> Dict[str, float]:
    """
    Compute all 4 metrics for mask-based evaluation.
    
    Args:
        gt_folder: Path to ground truth masks folder
        pred_folder: Path to prediction masks folder
    
    Returns:
        Dictionary containing all metric scores
    """
    print("Loading mask data...")
    
    # Load all masks
    gt_masks_list = []
    pred_masks_list = []
    
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
    
    print(f"Loaded {len(gt_masks_list)} videos for evaluation")
    
    # Calculate metrics
    metrics = OrderedDict()
    
    print("Calculating Temporal AP...")
    temporal_metric = MaskBasedTemporalDetection(gt_masks_list, pred_masks_list)
    temporal_metric.evaluate()
    metrics.update(temporal_metric.get_metrics())
    
    print("Calculating SpatioTemporal AP...")
    spatio_temporal_metric = MaskBasedSpatioTemporalDetection(gt_masks_list, pred_masks_list)
    spatio_temporal_metric.evaluate()
    metrics.update(spatio_temporal_metric.get_metrics())
    
    print("Calculating Success Metrics...")
    success_metric = MaskBasedSuccessMetrics(gt_masks_list, pred_masks_list)
    success_metric.evaluate()
    metrics.update(success_metric.get_metrics())
    
    print("Calculating Tracking Metrics...")
    tracking_metric = MaskBasedTrackingMetrics(gt_masks_list, pred_masks_list)
    tracking_metric.evaluate()
    metrics.update(tracking_metric.get_metrics())
    
    return metrics
