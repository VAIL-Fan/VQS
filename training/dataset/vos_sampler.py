# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class RandomUniformSampler(VOSSampler):
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
    ):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob

    def sample(self, video, segment_loader, epoch=None):

        for retry in range(MAX_RETRIES):
            # Need at least 8 frames: 1 query + 7 consecutive video frames
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )
            
            # Select 1 query frame (first frame) + 7 consecutive video frames
            # Query frame is randomly selected from available frames
            # Then select 7 consecutive frames starting from a random position
            max_start_idx = len(video.frames) - (self.num_frames - 1)  # -1 for query frame
            if max_start_idx <= 0:
                # Not enough frames for consecutive selection, use first frame as query and next 7
                query_idx = 0
                start_idx = 1
            else:
                query_idx = random.randrange(0, len(video.frames))
                # Select start position for 7 consecutive frames (avoid overlapping with query if possible)
                # If query is near the end, start from beginning; otherwise start after query
                if query_idx >= max_start_idx:
                    # Query is too close to end, start from beginning
                    start_idx = random.randrange(0, max_start_idx)
                else:
                    # Start after query frame, but ensure we have enough frames
                    if query_idx + 1 < max_start_idx:
                        start_idx = random.randrange(query_idx + 1, max_start_idx)
                    else:
                        start_idx = random.randrange(0, max_start_idx)
            
            query_frame = video.frames[query_idx]
            
            # Get 7 consecutive frames
            consecutive_frames = [video.frames[start_idx + i] for i in range(self.num_frames - 1)]
            
            # Create frames list: query frame (frame 0) + 7 consecutive frames (frames 1-7)
            frames = [query_frame] + consecutive_frames
            
            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]

            # Get first frame object ids
            visible_object_ids = []
            loaded_segms = segment_loader.load(frames[0].frame_idx)
            if isinstance(loaded_segms, LazySegments):
                # LazySegments for SA1BRawDataset
                visible_object_ids = list(loaded_segms.keys())
            else:
                for object_id, segment in segment_loader.load(
                    frames[0].frame_idx
                ).items():
                    if segment.sum():
                        visible_object_ids.append(object_id)

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class EvalSampler(VOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames
        object_ids = segment_loader.load(frames[0].frame_idx).keys()
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
