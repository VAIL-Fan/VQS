import os
import glob
from typing import List, Dict, Tuple
import numpy as np
import cv2
from collections import defaultdict

try:
    from .utils import BBox, ResponseTrack
except ImportError:
    try:
        from evaluation.structures import BBox, ResponseTrack
    except ImportError:
        from utils import BBox, ResponseTrack


def load_mask_from_path(mask_path: str) -> np.ndarray:
    """Load a binary mask from PNG file."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    return mask


def mask_to_bbox(mask: np.ndarray) -> BBox:
    """Convert binary mask to bounding box."""
    # Find non-zero pixels
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        # Empty mask - return empty bbox
        return BBox(0, 0, 0, 0, 0)
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return BBox(x_min, y_min, x_max, y_max, 0)  # frame number will be set later


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


def masks_to_response_track(masks: Dict[int, np.ndarray], video_id: str = "") -> ResponseTrack:
    """
    Convert dictionary of masks to ResponseTrack format.
    
    Args:
        masks: Dictionary mapping frame numbers to mask arrays
        video_id: Video identifier
    
    Returns:
        ResponseTrack object
    """
    bboxes = []
    frame_numbers = sorted(masks.keys())
    
    if not frame_numbers:
        # Empty track
        return ResponseTrack([], (0, 0), video_id)
    
    # Create temporal extent
    temporal_extent = (frame_numbers[0], frame_numbers[-1])
    
    # Convert each mask to bbox
    for frame_num in frame_numbers:
        mask = masks[frame_num]
        bbox = mask_to_bbox(mask)
        bbox.fno = frame_num
        bboxes.append(bbox)
    
    return ResponseTrack(bboxes, temporal_extent, video_id)


def load_all_video_masks(gt_folder: str, pred_folder: str) -> Tuple[List[ResponseTrack], List[List[ResponseTrack]]]:
    """
    Load ground truth and prediction masks for all videos.
    
    Args:
        gt_folder: Path to ground truth masks folder
        pred_folder: Path to prediction masks folder
    
    Returns:
        Tuple of (ground_truth_tracks, prediction_tracks)
    """
    # Get all video folders
    gt_videos = [d for d in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, d))]
    pred_videos = [d for d in os.listdir(pred_folder) if os.path.isdir(os.path.join(pred_folder, d))]
    
    # Find common videos
    common_videos = set(gt_videos) & set(pred_videos)
    print(f"Found {len(common_videos)} common videos")
    
    ground_truth_tracks = []
    prediction_tracks = []
    
    for video_id in sorted(common_videos):
        print(f"Processing video: {video_id}")
        
        # Load ground truth masks
        gt_folder_path = os.path.join(gt_folder, video_id)
        gt_masks = load_video_masks(gt_folder_path, exclude_query_frames=True)
        
        # Load prediction masks
        pred_folder_path = os.path.join(pred_folder, video_id)
        pred_masks = load_video_masks(pred_folder_path, exclude_query_frames=True)
        
        # Convert to ResponseTrack
        if gt_masks:  # Only process if ground truth has masks
            gt_track = masks_to_response_track(gt_masks, video_id)
            ground_truth_tracks.append(gt_track)
            
            # For predictions, create a list of tracks (in case there are multiple predictions)
            # For now, we'll create one track per video
            if pred_masks:
                pred_track = masks_to_response_track(pred_masks, video_id)
                prediction_tracks.append([pred_track])  # List of tracks for this video
            else:
                prediction_tracks.append([])  # No predictions for this video
        else:
            print(f"Warning: No ground truth masks found for {video_id}")
    
    print(f"Loaded {len(ground_truth_tracks)} ground truth tracks")
    print(f"Loaded {len(prediction_tracks)} prediction track sets")
    
    return ground_truth_tracks, prediction_tracks


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


def masks_to_spatio_temporal_iou(gt_masks: Dict[int, np.ndarray], 
                                pred_masks: Dict[int, np.ndarray]) -> float:
    """
    Calculate spatio-temporal IoU between ground truth and prediction masks.
    
    Args:
        gt_masks: Ground truth masks dictionary
        pred_masks: Prediction masks dictionary
    
    Returns:
        Spatio-temporal IoU score
    """
    # Find common frames
    common_frames = set(gt_masks.keys()) & set(pred_masks.keys())
    
    if not common_frames:
        return 0.0
    
    total_intersection = 0
    total_union = 0
    
    for frame_num in common_frames:
        gt_mask = gt_masks[frame_num]
        pred_mask = pred_masks[frame_num]
        
        # Ensure masks have same shape
        if gt_mask.shape != pred_mask.shape:
            # Resize pred_mask to match gt_mask
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # Calculate IoU for this frame
        mask_iou = calculate_mask_iou(gt_mask, pred_mask)
        
        # Calculate intersection and union areas
        gt_area = (gt_mask > 0).sum()
        pred_area = (pred_mask > 0).sum()
        intersection_area = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
        union_area = gt_area + pred_area - intersection_area
        
        total_intersection += intersection_area
        total_union += union_area
    
    if total_union == 0:
        return 0.0
    
    return total_intersection / total_union


def calculate_mask_temporal_iou(gt_masks: Dict[int, np.ndarray], 
                               pred_masks: Dict[int, np.ndarray]) -> float:
    """
    Calculate temporal IoU between ground truth and prediction masks.
    
    Args:
        gt_masks: Ground truth masks dictionary
        pred_masks: Prediction masks dictionary
    
    Returns:
        Temporal IoU score
    """
    gt_frames = set(gt_masks.keys())
    pred_frames = set(pred_masks.keys())
    
    intersection_frames = len(gt_frames & pred_frames)
    union_frames = len(gt_frames | pred_frames)
    
    if union_frames == 0:
        return 0.0
    
    return intersection_frames / union_frames
