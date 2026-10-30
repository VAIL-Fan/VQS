"""
Data structures for video segmentation evaluation.
"""

from typing import List, Tuple


class BBox:
    """Bounding box representation."""
    
    def __init__(self, x1: int, y1: int, x2: int, y2: int, fno: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.fno = fno  # frame number
    
    def area(self) -> float:
        """Calculate area of bounding box."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    def __repr__(self):
        return f"BBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, fno={self.fno})"


class ResponseTrack:
    """Response track containing bounding boxes across frames."""
    
    def __init__(self, bboxes: List[BBox], temporal_extent: Tuple[int, int], video_id: str = "", score: float = 1.0):
        self.bboxes = bboxes
        self.temporal_extent = temporal_extent  # (start_frame, end_frame)
        self.video_id = video_id
        self.score = score
    
    def length(self) -> int:
        """Get number of frames in the track."""
        return len(self.bboxes)
    
    def volume(self) -> float:
        """Calculate total volume (sum of all bbox areas)."""
        return sum(bbox.area() for bbox in self.bboxes)
    
    def has_score(self) -> bool:
        """Check if track has a score."""
        return hasattr(self, 'score') and self.score is not None
    
    def __repr__(self):
        return f"ResponseTrack(video_id='{self.video_id}', frames={self.length()}, extent={self.temporal_extent})"
