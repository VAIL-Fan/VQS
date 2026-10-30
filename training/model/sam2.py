# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)
from sam2.modeling.spatio_temporal_transformer import SpatioTemporalTransformer

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class AdaptiveGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Reduction MLPs for recent and distractor (variable slots -> 1 slot representation)
        self.recent_reducer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.distractor_reducer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # Final MLP to produce 3 adaptive weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
            nn.Sigmoid()  # Weights in [0, 1]
        )

    def forward(self, query_feats, recent_feats_list, distractor_feats_list):
        # query_feats: [1, C, H, W]
        # recent_feats_list: list of [1, C, H, W]
        # distractor_feats_list: list of [1, C, H, W]
        
        # 1. Global spatial pooling for all
        q_v = query_feats.mean(dim=(2, 3))  # [1, C]
        
        if len(recent_feats_list) > 0:
            # Pool each slot and aggregate (mean)
            r_v_all = torch.cat([f.mean(dim=(2, 3)) for f in recent_feats_list], dim=0)  # [N_r, C]
            r_v = r_v_all.mean(dim=0, keepdim=True)  # [1, C]
            r_v = self.recent_reducer(r_v)
        else:
            r_v = torch.zeros_like(q_v)
            
        if len(distractor_feats_list) > 0:
            # Pool each slot and aggregate (mean)
            d_v_all = torch.cat([f.mean(dim=(2, 3)) for f in distractor_feats_list], dim=0)  # [N_d, C]
            d_v = d_v_all.mean(dim=0, keepdim=True)  # [1, C]
            d_v = self.distractor_reducer(d_v)
        else:
            d_v = torch.zeros_like(q_v)
            
        # 2. Concatenate representations
        combined = torch.cat([q_v, r_v, d_v], dim=-1)  # [1, 3*C]
        
        # 3. Compute weights
        weights = self.gate_mlp(combined)  # [1, 3]
        return weights[0]  # [3] -> (w_q, w_r, w_d)


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        spatio_temporal_transformer=None,  # Optional spatio-temporal transformer
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        enable_two_stage_memory=True,
        two_stage_checkpointing=True,
        two_stage_iou_threshold=0.5,
        two_stage_topk=2,
        # Two-stage refinement options
        # - "pix": build stage-2 memory from `pix_feat_with_mem` (project+merge conv)
        # - "maskmem": build stage-2 memory directly from `maskmem_features` (recommended)
        two_stage_feature_source: str = "maskmem",
        # whether to weight selected frames' features by their stage-1 confidence scores
        two_stage_score_weighting: bool = False,
        # detach scores so stage-2 loss can't train the score head to inflate confidence
        two_stage_detach_scores: bool = True,
        # normalize weights across selected frames (only relevant if score weighting is enabled)
        two_stage_normalize_weights: bool = True,
        # Distractor-aware memory (DAM-style)
        enable_distractor_memory: bool = False,
        # Use GT distractor masks (union of other objects in the same video+frame).
        # Supported:
        # - "gt": union(other objects in same video+frame) on the query frame
        # - "pred_top1_low_gt": from stage-1 query multimask outputs, pick top-1 high-confidence
        #   candidate with low GT IoU (training only), then encode as distractor for stage 2
        # - "pred_top1_low_pred": (inference-capable) from stage-1 multimask outputs, pick top-1
        #   competing mask with high confidence but low overlap with the best predicted mask
        distractor_memory_source: str = "gt",
        # Whether to append distractor memory to the memory bank in stage 1 / stage 2.
        distractor_use_in_stage1: bool = True,
        distractor_use_in_stage2: bool = True,
        # Predicted-distractor selection params (only used when source is pred_top1_low_gt)
        distractor_pred_min_score: float = 0.7,
        distractor_pred_max_gt_iou: float = 0.2,
        # for pred_top1_low_pred: require IoU(best, alt) <= (1 - distractor_pred_min_divergence)
        distractor_pred_min_divergence: float = 0.5,
        distractor_pred_mask_thresh: float = 0.0,  # threshold on logits for binarization
        # DAM4SAM-style RAM/DRM buffers (built for stage 2)
        enable_dam: bool = False,
        dam_ram_size: int = 2,
        dam_drm_size: int = 1,
        dam_ram_min_score: float = 0.5,
        dam_drm_selection: str = "recent",  # {"recent","max_divergence"}
        dam_drm_strength_mode: str = "divergence",  # {"divergence","score_weighted"}
        **kwargs,
    ):
        # Extract attention gating parameters before passing to parent
        enable_attention_gating = kwargs.pop("enable_attention_gating", False)
        gate_query_init = kwargs.pop("gate_query_init", 1.0)
        gate_ram_init = kwargs.pop("gate_ram_init", 1.0)
        gate_drm_init = kwargs.pop("gate_drm_init", 0.5)

        # Extract adaptive gating parameter
        enable_adaptive_gating = kwargs.pop("enable_adaptive_gating", False)

        # Extract quality-weighted RAM selection parameters before passing to parent
        enable_quality_weighted_ram = kwargs.pop("enable_quality_weighted_ram", False)
        ram_selection_alpha = float(kwargs.pop("ram_selection_alpha", 0.4))
        ram_recency_decay = float(kwargs.pop("ram_recency_decay", 0.2))
        
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        
        # Spatio-temporal transformer (optional)
        self.spatio_temporal_transformer = spatio_temporal_transformer
        if self.spatio_temporal_transformer is not None:
            logging.info("Spatio-temporal transformer enabled for video frames")

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # Two-stage memory refinement
        self.enable_two_stage_memory = enable_two_stage_memory
        self.two_stage_checkpointing = two_stage_checkpointing and enable_two_stage_memory
        self.two_stage_iou_threshold = two_stage_iou_threshold
        self.two_stage_topk = max(1, two_stage_topk)
        self.two_stage_feature_source = two_stage_feature_source
        self.two_stage_score_weighting = two_stage_score_weighting
        self.two_stage_detach_scores = two_stage_detach_scores
        self.two_stage_normalize_weights = two_stage_normalize_weights

        # Distractor-aware memory
        self.enable_distractor_memory = enable_distractor_memory
        self.distractor_memory_source = distractor_memory_source
        self.distractor_use_in_stage1 = distractor_use_in_stage1
        self.distractor_use_in_stage2 = distractor_use_in_stage2
        self.distractor_pred_min_score = distractor_pred_min_score
        self.distractor_pred_max_gt_iou = distractor_pred_max_gt_iou
        self.distractor_pred_min_divergence = distractor_pred_min_divergence
        self.distractor_pred_mask_thresh = distractor_pred_mask_thresh

        self.enable_dam = enable_dam
        self.dam_ram_size = max(0, int(dam_ram_size))
        self.dam_drm_size = max(0, int(dam_drm_size))
        self.dam_ram_min_score = float(dam_ram_min_score)
        self.dam_drm_selection = dam_drm_selection
        self.dam_drm_strength_mode = dam_drm_strength_mode

        # Quality-weighted RAM selection (optional)
        # When enabled, Stage 1 IoU scores are used for both filtering and ranking,
        # combined with a recency weight. When disabled, the original greedy
        # recency-based selection is used.
        self.enable_quality_weighted_ram = enable_quality_weighted_ram
        self.ram_selection_alpha = ram_selection_alpha  # weight for recency vs quality
        self.ram_recency_decay = ram_recency_decay  # decay rate for recency weights
        
        # Attention gating for RAM/DRM (explicit control over memory contribution)
        self.enable_attention_gating = enable_attention_gating
        if self.enable_attention_gating:
            # Learnable gates: positive for Query/RAM (attend), can be negative for DRM (suppress)
            self.gate_query = nn.Parameter(torch.tensor(gate_query_init))
            self.gate_ram = nn.Parameter(torch.tensor(gate_ram_init))
            self.gate_drm = nn.Parameter(torch.tensor(gate_drm_init))
            logging.info(
                f"Attention gating enabled: gate_query={gate_query_init}, "
                f"gate_ram={gate_ram_init}, gate_drm={gate_drm_init}"
            )
        else:
            self.gate_query = None
            self.gate_ram = None
            self.gate_drm = None
        
        # Adaptive gating: Input-dependent control over Query/RAM/DRM contribution
        self.enable_adaptive_gating = enable_adaptive_gating
        if self.enable_adaptive_gating:
            self.adaptive_gate = AdaptiveGate(self.mem_dim)
            logging.info("Adaptive gating enabled: input-dependent memory weights")
        else:
            self.adaptive_gate = None
        
        if self.enable_distractor_memory:
            # A learnable tag added to distractor memory features to separate them
            # from normal target memory features in the memory bank.
            self.distractor_embed_spatial = nn.Parameter(torch.zeros(self.mem_dim))
        else:
            self.distractor_embed_spatial = None
        if self.enable_two_stage_memory:
            if self.two_stage_feature_source == "pix":
                self.two_stage_frame_adapter = nn.Sequential(
                    nn.Conv2d(self.hidden_dim, self.mem_dim, kernel_size=1),
                    nn.SiLU(),
                )
                self.two_stage_merge_conv = nn.Sequential(
                    nn.Conv2d(
                        self.mem_dim * self.two_stage_topk, self.mem_dim, kernel_size=1
                    ),
                    nn.SiLU(),
                    nn.Conv2d(self.mem_dim, self.mem_dim, kernel_size=3, padding=1),
                )
            elif self.two_stage_feature_source == "maskmem":
                # No adapter/merge conv needed; we merge selected frames' maskmem features directly.
                self.two_stage_frame_adapter = None
                self.two_stage_merge_conv = None
            else:
                raise ValueError(
                    f"Unsupported two_stage_feature_source={self.two_stage_feature_source!r} "
                    f"(expected 'pix' or 'maskmem')"
                )
        else:
            self.two_stage_frame_adapter = None
            self.two_stage_merge_conv = None

    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)

        if not self.enable_two_stage_memory:
            return self.forward_tracking(backbone_out, input)

        stage1_outputs = self.forward_tracking(
            backbone_out,
            input,
            return_dict=True,
        )
        # Predicted-distractor DAM (training-only): select a competing mask from stage-1 query outputs
        # and encode it as an extra distractor memory slot for stage 2.
        if self.enable_distractor_memory and self.distractor_memory_source in (
            "pred_top1_low_gt",
            "pred_top1_low_pred",
        ):
            if self.distractor_memory_source == "pred_top1_low_gt" and not self.training:
                raise ValueError(
                    "distractor_memory_source='pred_top1_low_gt' requires GT masks and is only supported in training."
                )
            q0 = stage1_outputs["cond_frame_outputs"].get(0, None)
            if q0 is not None:
                if self.distractor_memory_source == "pred_top1_low_gt":
                    gt0 = backbone_out.get("gt_masks_per_frame", {}).get(0, None)  # [B,1,H,W]
                    dmask, has = self._select_predicted_distractor_mask_top1_low_gt(q0, gt0)
                else:
                    dmask, has = self._select_predicted_distractor_mask_top1_low_pred(q0)
                if dmask is not None and has is not None and bool(has.any().item()):
                    cached = stage1_outputs.get("backbone_cache", None)
                    if cached is not None and cached.get("vision_feats") is not None:
                        img_ids = input.flat_obj_to_img_idx[0]
                        q_vision_feats = [x[:, img_ids] for x in cached["vision_feats"]]
                        q_feat_sizes = cached["feat_sizes"]
                        d_feats, d_pos = self._encode_external_memory_from_mask(
                            current_vision_feats=q_vision_feats[-1:],
                            feat_sizes=q_feat_sizes[-1:],
                            mask_for_mem=dmask,
                        )
                        if self.distractor_embed_spatial is not None and d_feats.dim() == 4:
                            d_feats = d_feats + self.distractor_embed_spatial.view(1, -1, 1, 1)
                        q0["distractor_maskmem_features"] = d_feats
                        q0["distractor_maskmem_pos_enc"] = d_pos
        stage2_outputs = self._run_second_stage(backbone_out, input, stage1_outputs)
        
        # If in training mode, return both stage 1 and stage 2 outputs to combine losses
        if self.training:
            # Return a list or dict that the loss function can handle
            # Flatten stage 1 outputs similar to stage 2
            stage1_flat = self._flatten_tracking_outputs(stage1_outputs)
            
            # Combine stage 1 and stage 2 outputs into a list of per-frame dictionaries
            combined_outputs = []
            # Assuming stage1_flat and stage2_outputs are lists of equal length (num_frames)
            # where each element corresponds to a frame's output
            for s1, s2 in zip(stage1_flat, stage2_outputs):
                combined_outputs.append({
                    "stage1": s1,
                    "stage2": s2
                })
            return combined_outputs
            
        return stage2_outputs

    def _get_stage1_frame_scores(self, frame_out: dict) -> torch.Tensor | None:
        """Return per-object confidence score [B] from stage-1 outputs."""
        if frame_out is None:
            return None
        ious_list = frame_out.get("multistep_pred_ious", None)
        if not ious_list:
            return None
        ious = ious_list[0]  # [B, M]
        if ious is None:
            return None
        return ious.max(dim=-1).values  # [B]

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # Distractor masks per object per frame (union of other objects in same video+frame)
        # Shape: [O, 1, H, W] for each t, aligned with `input.masks[t]` ordering.
        distractor_masks_per_frame = {}
        if self.enable_distractor_memory and self.distractor_memory_source == "gt":
            for t, masks_t in enumerate(input.masks):
                # masks_t: [O, H, W] (bool)
                ids_t = input.metadata.unique_objects_identifier[t]  # [O, 3] = (video_id, obj_id, frame_id)
                video_ids = ids_t[:, 0]
                # Map arbitrary video_id -> contiguous group id
                _, inv = torch.unique(video_ids, return_inverse=True)
                masks_f = masks_t.to(torch.float32)
                O, Hm, Wm = masks_f.shape
                group_sum = torch.zeros(
                    int(inv.max().item()) + 1, Hm, Wm, device=masks_f.device, dtype=masks_f.dtype
                )
                group_sum.scatter_add_(
                    0, inv.view(O, 1, 1).expand(O, Hm, Wm), masks_f
                )
                sum_for_obj = group_sum[inv]  # [O, H, W]
                distractor = (sum_for_obj - masks_f) > 0.0
                distractor_masks_per_frame[t] = distractor.unsqueeze(1).to(torch.float32)
        backbone_out["distractor_masks_per_frame"] = distractor_masks_per_frame
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def _encode_external_memory_from_mask(self, current_vision_feats, feat_sizes, mask_for_mem: torch.Tensor):
        """
        Encode an externally provided mask into SAM2 memory feature space using the
        existing memory encoder. `mask_for_mem` is expected to be in [0,1] already
        (e.g., binary GT mask), shape [B, 1, H_img, W_img] matching the SAM head output resolution.
        """
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        mask_for_mem = mask_for_mem.to(dtype=pix_feat.dtype, device=pix_feat.device)
        # Apply same scaling/bias as SAM2 memory encoding expects.
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        feats = maskmem_out["vision_features"]
        pos = maskmem_out["vision_pos_enc"]
        return feats, pos

    def _get_query_memory_bank(self, query_output: dict, *, use_distractor: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (memory_feats, memory_pos) as 5D tensors [B, M, C, H, W] ready for _flatten_memory_tokens.
        """
        feats = query_output["maskmem_features"]
        pos = query_output["maskmem_pos_enc"][-1]
        if feats.dim() == 4:
            feats = feats.unsqueeze(1)
        if pos.dim() == 4:
            pos = pos.unsqueeze(1)

        if use_distractor and self.enable_distractor_memory:
            d_feats = query_output.get("distractor_maskmem_features", None)
            d_pos_list = query_output.get("distractor_maskmem_pos_enc", None)
            d_pos = None if d_pos_list is None else d_pos_list[-1]
            if d_feats is not None and d_pos is not None:
                if d_feats.dim() == 4:
                    d_feats = d_feats.unsqueeze(1)
                if d_pos.dim() == 4:
                    d_pos = d_pos.unsqueeze(1)
                feats = torch.cat([feats, d_feats], dim=1)
                pos = torch.cat([pos, d_pos], dim=1)

        return feats, pos

    def _select_predicted_distractor_mask_top1_low_gt(
        self,
        query_out: dict,
        gt_masks: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Select a per-object distractor mask from stage-1 query outputs:
        - take the highest-confidence predicted mask (IoU head) among candidates
          whose GT IoU is <= distractor_pred_max_gt_iou and score >= distractor_pred_min_score.

        Args:
            query_out: output dict for query frame (frame 0) from stage 1.
            gt_masks: [B, 1, H, W] float/bool GT masks for each object.
        Returns:
            (distractor_mask, has):
              - distractor_mask: [B, 1, H, W] float mask in {0,1}
              - has: [B] bool indicating which objects have a valid distractor for this frame
            Returns (None, None) if no object has a valid distractor.
        """
        if "multistep_pred_multimasks_high_res" not in query_out:
            return None, None
        if "multistep_pred_ious" not in query_out:
            return None, None
        if not query_out["multistep_pred_multimasks_high_res"]:
            return None, None
        if not query_out["multistep_pred_ious"]:
            return None, None

        logits = query_out["multistep_pred_multimasks_high_res"][0]  # [B, M, H, W] (logits)
        scores = query_out["multistep_pred_ious"][0]  # [B, M]
        if logits is None or scores is None:
            return None, None
        if gt_masks is None:
            return None, None

        gt = gt_masks.to(dtype=torch.bool)
        B, M, H, W = logits.shape
        # binarize predicted masks using logits threshold (default 0 => prob 0.5)
        pred_bin = (logits > float(self.distractor_pred_mask_thresh))  # [B, M, H, W]

        # compute IoU(pred, gt) per candidate mask
        gt_b = gt.expand(B, M, H, W)
        inter = (pred_bin & gt_b).flatten(2).sum(dim=2).to(dtype=torch.float32)  # [B,M]
        union = (pred_bin | gt_b).flatten(2).sum(dim=2).to(dtype=torch.float32)  # [B,M]
        gt_iou = inter / (union + 1e-6)

        # valid candidates: high score and low GT IoU
        valid = (scores >= float(self.distractor_pred_min_score)) & (
            gt_iou <= float(self.distractor_pred_max_gt_iou)
        )

        # pick best score among valid; if none valid, fall back to "no distractor" for that object
        masked_scores = scores.clone()
        masked_scores[~valid] = -1e9
        best_idx = masked_scores.argmax(dim=1)  # [B]
        best_score = masked_scores.max(dim=1).values  # [B]
        has = best_score > -1e8
        if not bool(has.any().item()):
            return None, None

        out = torch.zeros((B, 1, H, W), device=logits.device, dtype=torch.float32)
        for b in range(B):
            if bool(has[b].item()):
                out[b, 0] = pred_bin[b, best_idx[b]].to(torch.float32)
        return out, has

    def _select_predicted_distractor_mask_top1_low_pred(
        self,
        frame_out: dict,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Inference-capable distractor selection based on divergence:
          divergence = 1 - IoU(best_mask, alt_mask)
        Select top-1 alt mask with:
          - score >= distractor_pred_min_score
          - divergence >= distractor_pred_min_divergence

        Returns (mask, has) similar to _select_predicted_distractor_mask_top1_low_gt.
        """
        if frame_out is None:
            return None, None
        if "multistep_pred_multimasks_high_res" not in frame_out:
            return None, None
        if "multistep_pred_ious" not in frame_out:
            return None, None
        if not frame_out["multistep_pred_multimasks_high_res"]:
            return None, None
        if not frame_out["multistep_pred_ious"]:
            return None, None

        logits = frame_out["multistep_pred_multimasks_high_res"][0]  # [B, M, H, W]
        scores = frame_out["multistep_pred_ious"][0]  # [B, M]
        if logits is None or scores is None:
            return None, None

        B, M, H, W = logits.shape
        if M <= 1:
            return None, None

        pred_bin = (logits > float(self.distractor_pred_mask_thresh))  # [B, M, H, W]

        best_idx = scores.argmax(dim=1)  # [B]
        # gather best masks
        best_masks = torch.zeros((B, H, W), device=logits.device, dtype=torch.bool)
        for b in range(B):
            best_masks[b] = pred_bin[b, best_idx[b]]

        # compute IoU(best, alt) for all candidates
        best_b = best_masks.unsqueeze(1).expand(B, M, H, W)
        inter = (pred_bin & best_b).flatten(2).sum(dim=2).to(dtype=torch.float32)  # [B,M]
        union = (pred_bin | best_b).flatten(2).sum(dim=2).to(dtype=torch.float32)  # [B,M]
        iou_best = inter / (union + 1e-6)
        divergence = 1.0 - iou_best

        valid = (scores >= float(self.distractor_pred_min_score)) & (
            divergence >= float(self.distractor_pred_min_divergence)
        )
        # exclude the best mask itself
        valid.scatter_(1, best_idx.view(B, 1), False)

        masked_scores = scores.clone()
        masked_scores[~valid] = -1e9
        alt_idx = masked_scores.argmax(dim=1)  # [B]
        alt_score = masked_scores.max(dim=1).values  # [B]
        has = alt_score > -1e8
        if not bool(has.any().item()):
            return None, None

        out = torch.zeros((B, 1, H, W), device=logits.device, dtype=torch.float32)
        for b in range(B):
            if bool(has[b].item()):
                out[b, 0] = pred_bin[b, alt_idx[b]].to(torch.float32)
        return out, has

    def _score_predicted_distractor_top1_low_gt(
        self, frame_out: dict, gt_masks: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Return (strength, has) for pred_top1_low_gt selection.
        strength = 1 - IoU(pred, GT) for the selected distractor candidate.
        """
        dmask, has = self._select_predicted_distractor_mask_top1_low_gt(frame_out, gt_masks)
        if dmask is None or has is None:
            return None, None
        # Recompute GT IoU for selected mask to obtain strength.
        logits = frame_out["multistep_pred_multimasks_high_res"][0]
        scores = frame_out["multistep_pred_ious"][0]
        pred_bin = (logits > float(self.distractor_pred_mask_thresh))
        gt = gt_masks.to(dtype=torch.bool)
        B, M, H, W = logits.shape
        gt_b = gt.expand(B, M, H, W)
        inter = (pred_bin & gt_b).flatten(2).sum(dim=2).to(dtype=torch.float32)
        union = (pred_bin | gt_b).flatten(2).sum(dim=2).to(dtype=torch.float32)
        gt_iou = inter / (union + 1e-6)
        valid = (scores >= float(self.distractor_pred_min_score)) & (
            gt_iou <= float(self.distractor_pred_max_gt_iou)
        )
        masked_scores = scores.clone()
        masked_scores[~valid] = -1e9
        best_idx = masked_scores.argmax(dim=1)
        best_gt_iou = gt_iou[torch.arange(B, device=gt_iou.device), best_idx]
        strength = (1.0 - best_gt_iou).to(torch.float32)
        if self.dam_drm_strength_mode == "score_weighted":
            best_score = scores[torch.arange(B, device=scores.device), best_idx].to(torch.float32)
            strength = strength * best_score
        elif self.dam_drm_strength_mode != "divergence":
            raise ValueError(
                f"Unsupported dam_drm_strength_mode={self.dam_drm_strength_mode!r} "
                f"(expected 'divergence' or 'score_weighted')"
            )
        strength = strength * has.to(torch.float32)
        return strength, has

    def _score_predicted_distractor_top1_low_pred(
        self, frame_out: dict
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Return (strength, has) for pred_top1_low_pred selection.
        strength = divergence = 1 - IoU(best, alt) for the selected alt.
        """
        if frame_out is None:
            return None, None
        if "multistep_pred_multimasks_high_res" not in frame_out:
            return None, None
        if "multistep_pred_ious" not in frame_out:
            return None, None
        if not frame_out["multistep_pred_multimasks_high_res"]:
            return None, None
        if not frame_out["multistep_pred_ious"]:
            return None, None

        logits = frame_out["multistep_pred_multimasks_high_res"][0]
        scores = frame_out["multistep_pred_ious"][0]
        B, M, H, W = logits.shape
        if M <= 1:
            return None, None
        pred_bin = (logits > float(self.distractor_pred_mask_thresh))
        best_idx = scores.argmax(dim=1)
        best_masks = torch.zeros((B, H, W), device=logits.device, dtype=torch.bool)
        for b in range(B):
            best_masks[b] = pred_bin[b, best_idx[b]]
        best_b = best_masks.unsqueeze(1).expand(B, M, H, W)
        inter = (pred_bin & best_b).flatten(2).sum(dim=2).to(dtype=torch.float32)
        union = (pred_bin | best_b).flatten(2).sum(dim=2).to(dtype=torch.float32)
        iou_best = inter / (union + 1e-6)
        divergence = 1.0 - iou_best
        valid = (scores >= float(self.distractor_pred_min_score)) & (
            divergence >= float(self.distractor_pred_min_divergence)
        )
        valid.scatter_(1, best_idx.view(B, 1), False)
        masked_scores = scores.clone()
        masked_scores[~valid] = -1e9
        alt_idx = masked_scores.argmax(dim=1)
        alt_score = masked_scores.max(dim=1).values
        has = alt_score > -1e8
        strength = divergence[torch.arange(B, device=divergence.device), alt_idx].to(torch.float32)
        if self.dam_drm_strength_mode == "score_weighted":
            sel_score = scores[torch.arange(B, device=scores.device), alt_idx].to(torch.float32)
            strength = strength * sel_score
        elif self.dam_drm_strength_mode != "divergence":
            raise ValueError(
                f"Unsupported dam_drm_strength_mode={self.dam_drm_strength_mode!r} "
                f"(expected 'divergence' or 'score_weighted')"
            )
        strength = strength * has.to(torch.float32)
        return strength, has

    def forward_tracking(
        self,
        backbone_out,
        input: BatchedVideoDatapoint,
        return_dict=False,
        cached_backbone=None,
        override_cond_outputs=None,
        checkpoint_sam=False,
    ):
        """
        Modified forward tracking: 
        - Frame 0 (query) is processed first to create memory
        - Frames 1-7 are processed in parallel, all using the same query memory
        """
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        vision_feats = vision_pos_embeds = feat_sizes = None
        if cached_backbone is not None:
            img_feats_already_computed = True
            vision_feats = cached_backbone.get("vision_feats")
            vision_pos_embeds = cached_backbone.get("vision_pos_embeds")
            feat_sizes = cached_backbone.get("feat_sizes")
        elif img_feats_already_computed:
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        output_dict["num_frames"] = num_frames
        # Whether to include distractor memory for this pass (stage 1 vs stage 2).
        output_dict["_use_distractor_memory"] = bool(
            self.enable_distractor_memory
            and (
                self.distractor_use_in_stage2
                if override_cond_outputs is not None
                else self.distractor_use_in_stage1
            )
        )
        
        if override_cond_outputs is None:
            # Step 1: Process frame 0 (query frame) first to create memory
            query_frame_idx = 0
            img_ids = input.flat_obj_to_img_idx[query_frame_idx]
            if img_feats_already_computed:
                query_vision_feats = [x[:, img_ids] for x in vision_feats]
                query_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                (
                    _,
                    query_vision_feats,
                    query_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )
            
            query_out = self.track_step(
                frame_idx=query_frame_idx,
                is_init_cond_frame=True,  # Query frame is always a conditioning frame
                current_vision_feats=query_vision_feats,
                current_vision_pos_embeds=query_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(query_frame_idx, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(query_frame_idx, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(query_frame_idx, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            # Optionally encode a GT-based distractor memory on the query frame (frame 0).
            # NOTE: predicted-distractor mode is handled after stage-1 forward (needs stage-1 masks).
            if self.enable_distractor_memory and self.distractor_memory_source == "gt":
                dmask = backbone_out.get("distractor_masks_per_frame", {}).get(
                    query_frame_idx, None
                )
                if dmask is not None:
                    d_feats, d_pos = self._encode_external_memory_from_mask(
                        current_vision_feats=query_vision_feats[-1:],
                        feat_sizes=feat_sizes[-1:],
                        mask_for_mem=dmask,
                    )
                    # Tag as distractor memory
                    if self.distractor_embed_spatial is not None:
                        if d_feats.dim() == 4:
                            d_feats = d_feats + self.distractor_embed_spatial.view(
                                1, -1, 1, 1
                            )
                        elif d_feats.dim() == 5:
                            d_feats = d_feats + self.distractor_embed_spatial.view(
                                1, 1, -1, 1, 1
                            )
                    query_out["distractor_maskmem_features"] = d_feats
                    query_out["distractor_maskmem_pos_enc"] = d_pos
            output_dict["cond_frame_outputs"][query_frame_idx] = query_out
        else:
            output_dict["cond_frame_outputs"][0] = override_cond_outputs
        
        # Step 2: Process frames 1..num_frames-1 in parallel, all using the same query memory
        video_frame_indices = list(range(1, num_frames))  # Frames 1..num_frames-1
        
        # Process all video frames in parallel (batched together)
        video_outputs = self._process_video_frames_parallel(
            video_frame_indices=video_frame_indices,
            input=input,
            backbone_out=backbone_out,
            vision_feats=vision_feats if img_feats_already_computed else None,
            vision_pos_embeds=vision_pos_embeds if img_feats_already_computed else None,
            feat_sizes=feat_sizes,
            img_feats_already_computed=img_feats_already_computed,
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            output_dict=output_dict,
            num_frames=num_frames,
            checkpoint_sam=checkpoint_sam,
        )
        
        # Store outputs in output_dict
        for frame_idx, current_out in zip(video_frame_indices, video_outputs):
            output_dict["non_cond_frame_outputs"][frame_idx] = current_out

        if return_dict:
            if img_feats_already_computed:
                output_dict["backbone_cache"] = {
                    "vision_feats": vision_feats,
                    "vision_pos_embeds": vision_pos_embeds,
                    "feat_sizes": feat_sizes,
                }
            else:
                output_dict["backbone_cache"] = None
            return output_dict

        return self._flatten_tracking_outputs(output_dict)

    def _process_video_frames_parallel(
        self,
        video_frame_indices,
        input,
        backbone_out,
        vision_feats,
        vision_pos_embeds,
        feat_sizes,
        img_feats_already_computed,
        frames_to_add_correction_pt,
        output_dict,
        num_frames,
        checkpoint_sam=False,
    ):
        """
        Process all video frames (1-7) in parallel, all using the same query memory.
        
        This method batches all video frames together for efficient parallel processing:
        1. Stacks all frame features along batch dimension: [HW, B*num_frames, C]
        2. Processes memory attention in parallel for all frames (most expensive operation)
        3. Each frame still processes SAM decoder separately (due to different point_inputs/mask_inputs)
        
        Key improvements:
        - Memory attention is now batched: all frames processed together
        - Query memory is expanded to all frames: [B, mem_dim, H', W'] -> [B*num_frames, mem_dim, H', W']
        - Temporal position encodings are applied per frame in the batched tensor
        
        Note: SAM decoder is still processed per-frame because each frame may have
        different point_inputs/mask_inputs. To fully parallelize SAM decoder, you would
        need to batch point_inputs and mask_inputs as well (requires padding/alignment).
        """
        if len(video_frame_indices) == 0:
            return []
        
        # Get image features for all video frames
        all_video_vision_feats = []
        all_video_vision_pos_embeds = []
        
        if img_feats_already_computed:
            for frame_idx in video_frame_indices:
                img_ids = input.flat_obj_to_img_idx[frame_idx]
                frame_vision_feats = [x[:, img_ids] for x in vision_feats]
                frame_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
                all_video_vision_feats.append(frame_vision_feats)
                all_video_vision_pos_embeds.append(frame_vision_pos_embeds)
        else:
            for frame_idx in video_frame_indices:
                img_ids = input.flat_obj_to_img_idx[frame_idx]
                (
                    _,
                    frame_vision_feats,
                    frame_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )
                all_video_vision_feats.append(frame_vision_feats)
                all_video_vision_pos_embeds.append(frame_vision_pos_embeds)
        
        # Batch all frames together: stack features along batch dimension
        # vision_feats format: List of [HW, B, C] tensors
        # We need to stack B dimensions across frames
        num_video_frames = len(video_frame_indices)
        B = all_video_vision_feats[0][-1].size(1)  # Batch size per frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = all_video_vision_feats[0][-1].device
        
        # Stack all frame features: [HW, B*num_frames, C]
        batched_vision_feats = []
        batched_vision_pos_embeds = []
        for level_idx in range(len(all_video_vision_feats[0])):
            # Stack features from all frames: [HW, B*num_frames, C]
            stacked_feats = torch.cat([feats[level_idx] for feats in all_video_vision_feats], dim=1)
            stacked_pos = torch.cat([pos[level_idx] for pos in all_video_vision_pos_embeds], dim=1)
            batched_vision_feats.append(stacked_feats)
            batched_vision_pos_embeds.append(stacked_pos)
        
        # Prepare memory-conditioned features for all frames in parallel
        # All frames use the same query memory
        query_output = output_dict["cond_frame_outputs"].get(0, None)
        if query_output is None:
            # Fallback: no memory
            if self.num_maskmem == 0:
                pix_feat = batched_vision_feats[-1].permute(1, 2, 0).view(B * num_video_frames, C, H, W)
            elif self.directly_add_no_mem_embed:
                pix_feat_with_mem = batched_vision_feats[-1] + self.no_mem_embed
                pix_feat = pix_feat_with_mem.permute(1, 2, 0).view(B * num_video_frames, C, H, W)
            else:
                # Use no_mem_embed
                to_cat_memory = [self.no_mem_embed.expand(1, B * num_video_frames, self.mem_dim)]
                to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B * num_video_frames, self.mem_dim)]
                memory = torch.cat(to_cat_memory, dim=0)
                memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
                # Memory attention expects a list of length 1, so we pass only the last (lowest resolution) FPN level
                pix_feat_with_mem = self.memory_attention(
                    curr=[batched_vision_feats[-1]],  # Pass as list of length 1
                    curr_pos=[batched_vision_pos_embeds[-1]],  # Pass as list of length 1
                    memory=memory,
                    memory_pos=memory_pos_embed,
                    num_obj_ptr_tokens=0,
                )
                pix_feat = pix_feat_with_mem.permute(1, 2, 0).view(B * num_video_frames, C, H, W)
        else:
            # Use query frame memory - expand to all batched frames
            # DAM variable-length memory bank: run memory attention per object (no padding noise).
            if self.enable_dam and "dam_memory_feats_per_obj" in query_output:
                feats_per_obj = query_output["dam_memory_feats_per_obj"]
                pos_per_obj = query_output["dam_memory_pos_per_obj"]
                pix_feat_with_mem_all = torch.zeros(
                    (B * num_video_frames, C, H, W),
                    device=device,
                    dtype=batched_vision_feats[-1].dtype,
                )
                for b in range(B):
                    idxs = torch.tensor(
                        [b + i * B for i in range(num_video_frames)],
                        device=device,
                        dtype=torch.long,
                    )
                    curr = [batched_vision_feats[-1][:, idxs]]
                    curr_pos = [batched_vision_pos_embeds[-1][:, idxs]]
                    mem_feats_b = feats_per_obj[b].unsqueeze(0)  # [1, M_b, C, H, W]
                    mem_pos_b = pos_per_obj[b].unsqueeze(0)  # [1, M_b, C, H, W]
                    memory_tokens, memory_pos_tokens = self._flatten_memory_tokens(
                        mem_feats_b,
                        mem_pos_b,
                        num_video_frames,
                        device,
                    )
                    mem = memory_tokens
                    mem_pos = memory_pos_tokens
                    pix_hw_bc = self.memory_attention(
                        curr=curr,
                        curr_pos=curr_pos,
                        memory=mem,
                        memory_pos=mem_pos,
                        num_obj_ptr_tokens=0,
                    )
                    pix_bchw = pix_hw_bc.permute(1, 2, 0).view(
                        num_video_frames, C, H, W
                    )
                    for i in range(num_video_frames):
                        pix_feat_with_mem_all[i * B + b] = pix_bchw[i]
                pix_feat = pix_feat_with_mem_all
                # Skip object pointers in DAM-per-object path (can be added later if needed)
                num_obj_ptr_tokens = 0
            else:
                use_distractor = bool(output_dict.get("_use_distractor_memory", False))
                mem_feats, mem_pos = self._get_query_memory_bank(
                    query_output, use_distractor=use_distractor
                )
                memory_tokens, memory_pos_tokens = self._flatten_memory_tokens(
                    mem_feats,
                    mem_pos,
                    num_video_frames,
                    device,
                )
                to_cat_memory = [memory_tokens]
                to_cat_memory_pos_embed = [memory_pos_tokens]

                # Add object pointers if enabled (non-DAM path)
                num_obj_ptr_tokens = 0
                if self.use_obj_ptrs_in_encoder and "obj_ptr" in query_output:
                    obj_ptrs = query_output["obj_ptr"]  # [B, C]
                    expanded_obj_ptrs = obj_ptrs.repeat_interleave(
                        num_video_frames, dim=0
                    )  # [B*T, C]
                    obj_ptrs = expanded_obj_ptrs.unsqueeze(0)  # [1, B*T, C]

                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diffs = torch.tensor(
                            video_frame_indices, device=device, dtype=torch.float32
                        )  # [T]
                        t_diff_max = self.max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = get_1d_sine_pe(t_diffs / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)  # [T, mem_dim]
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                        obj_pos = obj_pos.reshape(-1, self.mem_dim).unsqueeze(0)
                    else:
                        obj_pos = obj_ptrs.new_zeros(
                            1, B * num_video_frames, self.mem_dim
                        )

                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(
                            -1,
                            B * num_video_frames,
                            C // self.mem_dim,
                            self.mem_dim,
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]

                memory = torch.cat(to_cat_memory, dim=0)
                memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
                pix_feat_with_mem = self.memory_attention(
                    curr=[batched_vision_feats[-1]],
                    curr_pos=[batched_vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos_embed,
                    num_obj_ptr_tokens=num_obj_ptr_tokens,
                )
                pix_feat = pix_feat_with_mem.permute(1, 2, 0).view(
                    B * num_video_frames, C, H, W
                )
        
        # Apply spatio-temporal transformer if enabled
        # This allows frames 1-7 to attend to each other across temporal and spatial dimensions
        if self.spatio_temporal_transformer is not None:
            # Apply transformer: [B*T, C, H, W] -> [B*T, C, H, W]
            # Pass T explicitly for proper processing
            pix_feat = self.spatio_temporal_transformer(pix_feat, T=num_video_frames)
        
        # Prepare high-res features for all frames
        if len(batched_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(batched_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        
        # Process all frames through SAM decoder in parallel
        # Note: Each frame may have different point_inputs/mask_inputs, so we need to handle them separately
        # For now, we'll process them in a batched way if possible, otherwise fall back to sequential
        # This is a simplified version - you may need to adjust based on your specific needs
        
        video_outputs = []
        for i, frame_idx in enumerate(video_frame_indices):
            # Extract features for this frame from batched tensor
            frame_start = i * B
            frame_end = (i + 1) * B
            frame_pix_feat = pix_feat[frame_start:frame_end]  # [B, C, H, W]
            
            frame_high_res_features = None
            if high_res_features is not None:
                frame_high_res_features = [
                    feat[frame_start:frame_end] for feat in high_res_features
                ]
            
            # Get inputs for this frame
            point_inputs = backbone_out["point_inputs_per_frame"].get(frame_idx, None)
            mask_inputs = backbone_out["mask_inputs_per_frame"].get(frame_idx, None)
            gt_masks = backbone_out["gt_masks_per_frame"].get(frame_idx, None)
            
            # Forward through SAM decoder
            multimask_output = self._use_multimask(False, point_inputs)
            # NOTE: We intentionally avoid wrapping the SAM decoder with
            # torch.utils.checkpoint when using DDP, because reentrant
            # checkpointing over shared decoder weights across Stage 1 & 2
            # can trigger "Expected to mark a variable ready only once".
            # If memory becomes an issue again, prefer reducing sequence
            # length / resolution instead of checkpointing this block.
            sam_outputs = self._forward_sam_heads(
                backbone_features=frame_pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=frame_high_res_features,
                multimask_output=multimask_output,
            )
            
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = sam_outputs
            
            # Create output dict for this frame
            current_out = {
                "point_inputs": point_inputs,
                "mask_inputs": mask_inputs,
                "multistep_pred_masks": low_res_masks,
                "multistep_pred_masks_high_res": high_res_masks,
                "multistep_pred_multimasks": [low_res_multimasks],
                "multistep_pred_multimasks_high_res": [high_res_multimasks],
                "multistep_pred_ious": [ious],
                "multistep_point_inputs": [point_inputs],
                "multistep_object_score_logits": [object_score_logits],
                "pred_masks": low_res_masks,
                "pred_masks_high_res": high_res_masks,
                "obj_ptr": obj_ptr,
                "pix_feat_with_mem": frame_pix_feat,
            }
            
            # Handle correction points if needed
            if frame_idx in frames_to_add_correction_pt:
                point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                    is_init_cond_frame=False,
                    point_inputs=point_inputs,
                    gt_masks=gt_masks,
                    high_res_features=frame_high_res_features,
                    pix_feat_with_mem=frame_pix_feat,
                    low_res_multimasks=low_res_multimasks,
                    high_res_multimasks=high_res_multimasks,
                    ious=ious,
                    low_res_masks=low_res_masks,
                    high_res_masks=high_res_masks,
                    object_score_logits=object_score_logits,
                    current_out=current_out,
                )
                (
                    _,
                    _,
                    _,
                    low_res_masks,
                    high_res_masks,
                    obj_ptr,
                    object_score_logits,
                ) = final_sam_outputs
                current_out["pred_masks"] = low_res_masks
                current_out["pred_masks_high_res"] = high_res_masks
                current_out["obj_ptr"] = obj_ptr
            
            # Skip memory encoding for video frames (1-7)
            # They don't contribute to memory bank - only frame 0's memory is used
            # Just set dummy memory features to avoid errors
            dummy_mem_feats = torch.zeros(B, self.mem_dim, H, W, device=device)
            dummy_mem_pos_enc = self.memory_encoder.position_encoding(dummy_mem_feats).to(dummy_mem_feats.dtype)
            current_out["maskmem_features"] = dummy_mem_feats
            current_out["maskmem_pos_enc"] = [dummy_mem_pos_enc]
            
            video_outputs.append(current_out)
        
        return video_outputs

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """
        Override to ensure frames 1-7 only use query frame (frame 0) memory.
        They should not use memory from previous frames in the 1-7 sequence.
        """
        # For frames 1-7, only use query frame (frame 0) memory
        if not is_init_cond_frame and frame_idx > 0:
            # This is a video frame (1-7), only use query frame memory
            B = current_vision_feats[-1].size(1)
            C = self.hidden_dim
            H, W = feat_sizes[-1]
            device = current_vision_feats[-1].device
            
            if self.num_maskmem == 0:
                pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
                return pix_feat
            
            # Only use query frame (frame 0) memory
            to_cat_memory, to_cat_memory_pos_embed = [], []
            
            # Get query frame (frame 0) output
            query_output = output_dict["cond_frame_outputs"].get(0, None)
            if query_output is None:
                # Fallback to no memory if query frame not found
                if self.directly_add_no_mem_embed:
                    pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                    pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                    return pix_feat_with_mem
                to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
                to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
            else:
                # Use query frame memory only
                use_distractor = bool(output_dict.get("_use_distractor_memory", False))
                mem_feats, mem_pos = self._get_query_memory_bank(
                    query_output, use_distractor=use_distractor
                )
                memory_tokens, memory_pos_tokens = self._flatten_memory_tokens(
                    mem_feats,
                    mem_pos,
                    num_video_frames=1,
                    device=device,
                )
                to_cat_memory.append(memory_tokens)
                to_cat_memory_pos_embed.append(memory_pos_tokens)
                
                # Add object pointers from query frame if enabled
                num_obj_ptr_tokens = 0
                if self.use_obj_ptrs_in_encoder and "obj_ptr" in query_output:
                    obj_ptrs = query_output["obj_ptr"].unsqueeze(0)  # [1, B, C]
                    if self.add_tpos_enc_to_obj_ptrs:
                        # Temporal position encoding for query frame (distance = frame_idx)
                        t_diff = frame_idx
                        t_diff_max = self.max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor([t_diff]).to(device=device, non_blocking=True)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(1, B, self.mem_dim)
                    
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
            
            # Concatenate memories and forward through transformer
            memory = torch.cat(to_cat_memory, dim=0)
            memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
            pix_feat_with_mem = self.memory_attention(
                curr=current_vision_feats,
                curr_pos=current_vision_pos_embeds,
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem
        else:
            # For query frame (frame 0) or other conditioning frames, use original logic
            return super()._prepare_memory_conditioned_features(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                track_in_reverse,
            )

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )
        current_out["pix_feat_with_mem"] = pix_feat

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Only encode memory for query frame (frame 0)
        # Frames 1-7 should not encode memory since they all use the same query memory
        if frame_idx == 0 or is_init_cond_frame:
            # Finally run the memory encoder on the predicted mask to encode
            # it into a new memory feature (that can be used in future frames)
            self._encode_memory_in_output(
                current_vision_feats,
                feat_sizes,
                point_inputs,
                run_mem_encoder,
                high_res_masks,
                object_score_logits,
                current_out,
            )
        else:
            # For frames 1-7, skip memory encoding (they don't contribute to memory)
            # Just set empty memory features to avoid errors
            B = current_vision_feats[-1].size(1)
            C = self.hidden_dim
            H, W = feat_sizes[-1]
            device = current_vision_feats[-1].device
            # Create dummy memory features (won't be used since these frames don't go into memory)
            dummy_mem_feats = torch.zeros(B, self.mem_dim, H, W, device=device)
            dummy_mem_pos_enc = self.memory_encoder.position_encoding(dummy_mem_feats).to(dummy_mem_feats.dtype)
            current_out["maskmem_features"] = dummy_mem_feats
            current_out["maskmem_pos_enc"] = [dummy_mem_pos_enc]
        
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs

    def _flatten_tracking_outputs(self, output_dict):
        combined = {}
        combined.update(output_dict["cond_frame_outputs"])
        combined.update(output_dict["non_cond_frame_outputs"])
        ordered = [combined[t] for t in range(output_dict["num_frames"])]
        flattened = []
        for entry in ordered:
            cleaned = dict(entry)
            cleaned.pop("pix_feat_with_mem", None)
            flattened.append({k: v for k, v in cleaned.items() if k != "obj_ptr"})
        return flattened

    def _run_second_stage(self, backbone_out, input, stage1_dict):
        override_memory = self._build_stage_two_override_memory(stage1_dict, backbone_out, input)
        if override_memory is None:
            return self._flatten_tracking_outputs(stage1_dict)
        # If distractor memory is enabled, carry it into stage 2 as well.
        # (We keep the distractor encoded from the query frame; stage-2 override modifies
        # only the target maskmem slots.)
        if self.enable_distractor_memory and self.distractor_use_in_stage2:
            q0 = stage1_dict["cond_frame_outputs"].get(0, None)
            if q0 is not None:
                if "distractor_maskmem_features" in q0 and "distractor_maskmem_pos_enc" in q0:
                    override_memory["distractor_maskmem_features"] = q0["distractor_maskmem_features"]
                    override_memory["distractor_maskmem_pos_enc"] = q0["distractor_maskmem_pos_enc"]
        cached = stage1_dict.get("backbone_cache")
        stage2_dict = self.forward_tracking(
            backbone_out,
            input,
            return_dict=True,
            cached_backbone=cached,
            override_cond_outputs=override_memory,
            checkpoint_sam=self.two_stage_checkpointing,
        )
        return self._flatten_tracking_outputs(stage2_dict)

    def _build_stage_two_override_memory(self, stage1_dict, backbone_out, input):
        if not stage1_dict["non_cond_frame_outputs"]:
            return None
        query_out = stage1_dict["cond_frame_outputs"].get(0)
        if query_out is None or query_out.get("maskmem_features") is None:
            return None

        # DAM4SAM-style: build a memory bank for stage 2: [Query] + RAM slots + DRM slots.
        # DRM slots use "top-1 high confidence but low GT overlap" competing masks (training-only).
        if self.enable_dam:
            cached = stage1_dict.get("backbone_cache")
            if cached is None or cached.get("vision_feats") is None:
                return None

            # Query memory (target)
            q_mem = query_out["maskmem_features"]  # [B,C,H,W] or [B,M,C,H,W]
            q_pos = query_out["maskmem_pos_enc"][-1]
            if q_mem.dim() != 4 or q_pos.dim() != 4:
                # keep DAM simple for now (expects single-slot query mem)
                return None
            B, C, Hm, Wm = q_mem.shape

            # ----- RAM: recent reliable target memories from frames 1..T-1 -----
            # We build RAM/DRM per-object with variable slot counts (no padding).
            ram_frames_per_obj: list[list[int]] = [[] for _ in range(B)]
            if self.dam_ram_size > 0:
                # compute per-frame Stage 1 scores for each object
                frame_ids = sorted(stage1_dict["non_cond_frame_outputs"].keys())
                # build a [B, num_frames] score matrix
                scores = []
                for t in frame_ids:
                    s = self._get_stage1_frame_scores(stage1_dict["non_cond_frame_outputs"][t])
                    if s is None:
                        s = torch.zeros(B, device=q_mem.device, dtype=torch.float32)
                    scores.append(s)
                scores = torch.stack(scores, dim=1)  # [B, F] Stage-1 IoU scores

                if not self.enable_quality_weighted_ram:
                    # Original behavior: greedy recency-based selection.
                    # pick most recent frames with score >= dam_ram_min_score (per object)
                    eligible = scores >= self.dam_ram_min_score
                    used = torch.zeros_like(eligible)
                    for _ in range(self.dam_ram_size):
                        chosen = torch.full((B,), -1, device=q_mem.device, dtype=torch.long)
                        for b in range(B):
                            idxs = torch.where(eligible[b] & ~used[b])[0]
                            if idxs.numel() > 0:
                                # Greedy: pick the most recent eligible frame
                                chosen[b] = idxs.max()
                                used[b, chosen[b]] = True
                        for b in range(B):
                            if chosen[b] >= 0:
                                ram_frames_per_obj[b].append(
                                    frame_ids[int(chosen[b].item())]
                                )
                else:
                    # Quality-weighted selection:
                    # 1) Filter by dam_ram_min_score (same as current)
                    # 2) Normalize quality scores per object across eligible frames
                    # 3) Compute recency weights (exponential decay)
                    # 4) Combine quality + recency
                    # 5) Select top-K frames per object by combined score

                    B_, F = scores.shape
                    device = q_mem.device

                    # Step 1: eligibility mask (same threshold as current implementation)
                    eligible = scores >= self.dam_ram_min_score  # [B, F]

                    # Step 2: normalize quality scores per object on eligible frames
                    quality_scores = scores  # [B, F]
                    quality_normalized = torch.zeros_like(quality_scores)
                    for b in range(B_):
                        eligible_scores = quality_scores[b][eligible[b]]
                        if eligible_scores.numel() == 0:
                            continue
                        q_min = eligible_scores.min()
                        q_max = eligible_scores.max()
                        if q_max > q_min:
                            quality_normalized[b] = (quality_scores[b] - q_min) / (
                                q_max - q_min + 1e-8
                            )
                        else:
                            # All eligible frames have the same score; treat them equally.
                            quality_normalized[b] = eligible[b].float()

                    # Step 3: compute recency weights (per frame, shared across objects)
                    frame_positions = torch.tensor(frame_ids, device=device, dtype=torch.float32)  # [F]
                    T = frame_positions.max()
                    lambda_decay = self.ram_recency_decay
                    recency_weights = torch.exp(-lambda_decay * (T - frame_positions))  # [F]
                    # normalize to [0, 1]
                    rec_min = recency_weights.min()
                    rec_max = recency_weights.max()
                    recency_normalized = (recency_weights - rec_min) / (rec_max - rec_min + 1e-8)  # [F]

                    # Step 4: combine recency and quality
                    alpha = self.ram_selection_alpha  # weight on recency
                    alpha = max(0.0, min(1.0, float(alpha)))
                    rec_broadcast = recency_normalized.unsqueeze(0).expand(B_, -1)  # [B, F]
                    selection_scores = alpha * rec_broadcast + (1.0 - alpha) * quality_normalized  # [B, F]

                    # Mask out ineligible frames
                    selection_scores = selection_scores.masked_fill(~eligible, -1e9)

                    # Step 5: select top-K frames per object by combined score
                    k = min(self.dam_ram_size, F)
                    if k > 0:
                        topk_scores, topk_indices = torch.topk(selection_scores, k=k, dim=1)
                        for b in range(B_):
                            for j in range(k):
                                if topk_scores[b, j] > -1e8:
                                    frame_idx = frame_ids[int(topk_indices[b, j].item())]
                                    ram_frames_per_obj[b].append(frame_idx)

            # ----- DRM: competing distractor memories from frames 1..T-1 -----
            drm_frames_per_obj: list[list[int]] = [[] for _ in range(B)]
            if self.dam_drm_size > 0:
                frame_ids = sorted(stage1_dict["non_cond_frame_outputs"].keys())
                # Score candidate frames per object, then select either "recent" or "max_divergence".
                strengths = torch.full(
                    (B, len(frame_ids)),
                    -1e9,
                    device=q_mem.device,
                    dtype=torch.float32,
                )
                has_mat = torch.zeros((B, len(frame_ids)), device=q_mem.device, dtype=torch.bool)
                for j, t in enumerate(frame_ids):
                    out = stage1_dict["non_cond_frame_outputs"][t]
                    if self.distractor_memory_source == "pred_top1_low_gt":
                        if not self.training:
                            return None
                        gt = backbone_out.get("gt_masks_per_frame", {}).get(t, None)
                        if gt is None:
                            continue
                        strength, has = self._score_predicted_distractor_top1_low_gt(out, gt)
                    else:
                        strength, has = self._score_predicted_distractor_top1_low_pred(out)
                    if strength is None or has is None:
                        continue
                    strengths[:, j] = strength
                    has_mat[:, j] = has

                if self.dam_drm_selection == "recent":
                    # pick most recent valid frames
                    used = torch.zeros_like(has_mat)
                    for _ in range(self.dam_drm_size):
                        chosen = torch.full((B,), -1, device=q_mem.device, dtype=torch.long)
                        for b in range(B):
                            idxs = torch.where(has_mat[b] & ~used[b])[0]
                            if idxs.numel() > 0:
                                chosen[b] = idxs.max()
                                used[b, chosen[b]] = True
                        for b in range(B):
                            if chosen[b] >= 0:
                                drm_frames_per_obj[b].append(frame_ids[int(chosen[b].item())])
                elif self.dam_drm_selection == "max_divergence":
                    # pick top-K frames by strength (divergence or 1-gt_iou)
                    k = min(self.dam_drm_size, len(frame_ids))
                    top_vals, top_idx = torch.topk(strengths, k=k, dim=1)
                    for b in range(B):
                        for j in range(k):
                            if top_vals[b, j] > -1e8:
                                drm_frames_per_obj[b].append(frame_ids[int(top_idx[b, j].item())])
                else:
                    raise ValueError(
                        f"Unsupported dam_drm_selection={self.dam_drm_selection!r} (expected 'recent' or 'max_divergence')"
                    )

            # Build per-object variable-length memory banks for stage 2.
            dam_feats_per_obj: list[torch.Tensor] = []
            dam_pos_per_obj: list[torch.Tensor] = []
            dam_memory_types_per_obj: list[list[str]] = []  # Track memory types for gating
            for b in range(B):
                feats_b = [q_mem[b : b + 1]]
                pos_b = [q_pos[b : b + 1]]
                memory_types_b = ["query"]  # Slot 0 is always Query
                
                # RAM slots (target memories)
                for t in ram_frames_per_obj[b]:
                    out = stage1_dict["non_cond_frame_outputs"][t]
                    mem = out.get("maskmem_features", None)
                    pos_list = out.get("maskmem_pos_enc", None)
                    pos_t = None if pos_list is None else pos_list[-1]
                    if mem is not None and pos_t is not None and mem.dim() == 4 and pos_t.dim() == 4:
                        feats_b.append(mem[b : b + 1])
                        pos_b.append(pos_t[b : b + 1])
                        memory_types_b.append("ram")
                
                # DRM slots (distractor memories) - only if divergence trigger fires.
                for t in drm_frames_per_obj[b]:
                    out = stage1_dict["non_cond_frame_outputs"][t]
                    if self.distractor_memory_source == "pred_top1_low_gt":
                        gt = backbone_out.get("gt_masks_per_frame", {}).get(t, None)
                        dmask, has = self._select_predicted_distractor_mask_top1_low_gt(out, gt)
                    else:
                        dmask, has = self._select_predicted_distractor_mask_top1_low_pred(out)
                    if dmask is None or has is None or not bool(has[b].item()):
                        continue  # <-- NO fallback slot; skip entirely
                    img_ids = input.flat_obj_to_img_idx[t]
                    t_vision_feats = [x[:, img_ids] for x in cached["vision_feats"]]
                    t_feat_sizes = cached["feat_sizes"]
                    d_feats, d_pos_list = self._encode_external_memory_from_mask(
                        current_vision_feats=t_vision_feats[-1:],
                        feat_sizes=t_feat_sizes[-1:],
                        mask_for_mem=dmask[b : b + 1],
                    )
                    if self.distractor_embed_spatial is not None and d_feats.dim() == 4:
                        d_feats = d_feats + self.distractor_embed_spatial.view(1, -1, 1, 1)
                    feats_b.append(d_feats)
                    dam_pos = d_pos_list[-1]
                    pos_b.append(dam_pos)
                    memory_types_b.append("drm")
                
                # Apply attention gating or adaptive gating if enabled
                if self.enable_adaptive_gating:
                    # Collect separated features for adaptive weighting
                    r_feats = [feats_b[i] for i, t in enumerate(memory_types_b) if t == "ram"]
                    d_feats_list = [feats_b[i] for i, t in enumerate(memory_types_b) if t == "drm"]
                    
                    # Compute weights (w_q, w_r, w_d) from [feats_b[0], RAM_list, DRM_list]
                    w_q, w_r, w_d = self.adaptive_gate(feats_b[0], r_feats, d_feats_list)
                    
                    feats_b_gated = []
                    for i, t in enumerate(memory_types_b):
                        if t == "query":
                            feats_b_gated.append(feats_b[i] * w_q)
                        elif t == "ram":
                            feats_b_gated.append(feats_b[i] * w_r)
                        elif t == "drm":
                            feats_b_gated.append(feats_b[i] * w_d)
                        else:
                            feats_b_gated.append(feats_b[i])
                    feats_b = feats_b_gated
                elif self.enable_attention_gating:
                    feats_b_gated = []
                    for slot_idx, mem_type in enumerate(memory_types_b):
                        if mem_type == "query":
                            gate = self.gate_query
                        elif mem_type == "ram":
                            gate = self.gate_ram
                        elif mem_type == "drm":
                            gate = self.gate_drm
                        else:
                            gate = 1.0  # Default: no gating
                        # Apply gate: multiply memory features by gate value
                        feats_b_gated.append(feats_b[slot_idx] * gate)
                    feats_b = feats_b_gated
                
                dam_feats_per_obj.append(torch.cat(feats_b, dim=0))  # [M_b, C, H, W]
                dam_pos_per_obj.append(torch.cat(pos_b, dim=0))  # [M_b, C, H, W]
                dam_memory_types_per_obj.append(memory_types_b)

            override_memory = dict(query_out)
            # Keep the standard keys for compatibility (query-only); DAM banks are stored separately.
            override_memory["maskmem_features"] = q_mem
            override_memory["maskmem_pos_enc"] = list(query_out["maskmem_pos_enc"])
            override_memory["dam_memory_feats_per_obj"] = dam_feats_per_obj
            override_memory["dam_memory_pos_per_obj"] = dam_pos_per_obj
            override_memory["dam_memory_types_per_obj"] = dam_memory_types_per_obj  # For debugging/visualization
            return override_memory

        selections = self._select_stage_two_frames(stage1_dict["non_cond_frame_outputs"])
        if selections is None:
            return None
        merged_feats, merged_pos = self._merge_stage_two_features(
            selections,
            stage1_dict["non_cond_frame_outputs"],
            query_out,
        )
        if merged_feats is None:
            return None
        query_mem = query_out["maskmem_features"]
        query_pos = query_out["maskmem_pos_enc"][-1]
        combined_features = torch.stack([query_mem, merged_feats], dim=1)
        combined_pos = torch.stack([query_pos, merged_pos], dim=1)

        override_memory = dict(query_out)
        override_memory["maskmem_features"] = combined_features
        pos_list = list(query_out["maskmem_pos_enc"])
        pos_list[-1] = combined_pos
        override_memory["maskmem_pos_enc"] = pos_list
        return override_memory

    def _select_stage_two_frames(self, video_outputs):
        frame_ids = sorted(video_outputs.keys())
        if not frame_ids:
            return None
        sample_scores = []
        for idx in frame_ids:
            ious = video_outputs[idx]["multistep_pred_ious"][0]
            frame_score = ious.max(dim=-1).values
            sample_scores.append(frame_score)
        scores = torch.stack(sample_scores, dim=1)
        k = min(self.two_stage_topk, scores.size(1))
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)
        selections = []
        for b in range(scores.size(0)):
            per_sample = []
            for j in range(self.two_stage_topk):
                if j < k and topk_scores[b, j] > self.two_stage_iou_threshold:
                    frame_id = frame_ids[topk_indices[b, j].item()]
                    per_sample.append((frame_id, topk_scores[b, j]))
                else:
                    per_sample.append((None, None))
            selections.append(per_sample)
        return selections

    def _merge_stage_two_features(self, selections, video_outputs, query_out):
        query_mem = query_out.get("maskmem_features")
        query_pos = None
        if query_out.get("maskmem_pos_enc") is not None:
            query_pos = query_out["maskmem_pos_enc"][-1]
        if query_mem is None or query_pos is None:
            return None, None

        if self.two_stage_feature_source == "maskmem":
            # Merge selected frames' maskmem features directly (same space as the tracker expects).
            merged_feats = []
            merged_pos = []
            for b, per_sample in enumerate(selections):
                slot_feats = []
                slot_pos = []
                slot_scores = []
                for frame_id, score in per_sample:
                    if frame_id is None:
                        feat = query_mem[b : b + 1]
                        pos = query_pos[b : b + 1]
                        score_t = None
                    else:
                        out = video_outputs.get(frame_id, None)
                        feat = None if out is None else out.get("maskmem_features", None)
                        pos_enc_list = None if out is None else out.get("maskmem_pos_enc", None)
                        pos = None if pos_enc_list is None else pos_enc_list[-1]
                        if feat is None or pos is None:
                            # Fallback to query if missing
                            feat = query_mem[b : b + 1]
                            pos = query_pos[b : b + 1]
                            score_t = None
                        else:
                            feat = feat[b : b + 1]
                            pos = pos[b : b + 1]
                            score_t = score

                    slot_feats.append(feat)
                    slot_pos.append(pos)
                    slot_scores.append(score_t)

                # Build weights
                if self.two_stage_score_weighting:
                    weights = []
                    for s in slot_scores:
                        if s is None:
                            w = torch.ones((), device=query_mem.device, dtype=query_mem.dtype)
                        else:
                            w = s
                            if self.two_stage_detach_scores:
                                w = w.detach()
                            w = w.to(dtype=query_mem.dtype, device=query_mem.device)
                            w = torch.clamp(w, min=0.0)
                        weights.append(w)
                    weights = torch.stack(weights, dim=0)  # [K]
                    if self.two_stage_normalize_weights:
                        weights = weights / (weights.sum() + 1e-6)
                else:
                    weights = torch.full(
                        (len(slot_feats),),
                        1.0 / max(1, len(slot_feats)),
                        device=query_mem.device,
                        dtype=query_mem.dtype,
                    )

                # Weighted merge
                feat_stack = torch.cat(slot_feats, dim=0)  # [K, 1, C, H, W]
                pos_stack = torch.cat(slot_pos, dim=0)  # [K, 1, C, H, W]
                w = weights.view(-1, 1, 1, 1, 1)
                merged_feat = (feat_stack * w).sum(dim=0)  # [1, C, H, W]
                merged_pos_b = (pos_stack * w).sum(dim=0)  # [1, C, H, W]
                merged_feats.append(merged_feat)
                merged_pos.append(merged_pos_b)

            merged_feats = torch.cat(merged_feats, dim=0)  # [B, C, H, W]
            merged_pos = torch.cat(merged_pos, dim=0)  # [B, C, H, W]
            return merged_feats, merged_pos

        # Legacy: build stage-2 memory from pix_feat_with_mem (project + merge conv)
        if self.two_stage_frame_adapter is None or self.two_stage_merge_conv is None:
            return None, None
        # Use query's backbone feature (256-dim) for padding, to match adapter input
        query_pix = query_out.get("pix_feat_with_mem")
        if query_pix is None:
            return None, None
        target_hw = query_mem.shape[-2:]
        merged_feats = []
        merged_pos = []
        for b, per_sample in enumerate(selections):
            slot_feats = []
            slot_pos = []
            for frame_id, score in per_sample:
                if frame_id is None:
                    feat = self._project_stage_two_frame(query_pix[b : b + 1], target_hw)
                    pos = self.memory_encoder.position_encoding(feat).to(feat.dtype)
                    weight = None
                else:
                    frame_feat = video_outputs[frame_id]["pix_feat_with_mem"][b : b + 1]
                    feat = self._project_stage_two_frame(frame_feat, target_hw)
                    pos = self.memory_encoder.position_encoding(feat).to(feat.dtype)
                    weight = score
                    if self.two_stage_detach_scores and weight is not None:
                        weight = weight.detach()
                    if weight is not None and self.two_stage_score_weighting:
                        weight = torch.clamp(weight.to(feat.dtype), min=0.0).view(1, 1, 1, 1)
                if weight is not None and self.two_stage_score_weighting:
                    feat = feat * weight
                slot_feats.append(feat)
                slot_pos.append(pos)
            if len(slot_feats) < self.two_stage_topk:
                pad_count = self.two_stage_topk - len(slot_feats)
                feat_pad = self._project_stage_two_frame(query_pix[b : b + 1], target_hw)
                pos_pad = self.memory_encoder.position_encoding(feat_pad).to(feat_pad.dtype)
                for _ in range(pad_count):
                    slot_feats.append(feat_pad)
                    slot_pos.append(pos_pad)
            concat = torch.cat(slot_feats, dim=1)
            merged_feat = self.two_stage_merge_conv(concat)
            mean_pos = torch.stack(slot_pos, dim=0).mean(dim=0)
            merged_feats.append(merged_feat)
            merged_pos.append(mean_pos)
        merged_feats = torch.cat(merged_feats, dim=0)
        merged_pos = torch.cat(merged_pos, dim=0)
        return merged_feats, merged_pos

    def _project_stage_two_frame(self, frame_feat, target_hw):
        mem_feat = self.two_stage_frame_adapter(frame_feat)
        if mem_feat.shape[-2:] != target_hw:
            mem_feat = F.interpolate(
                mem_feat, size=target_hw, mode="bilinear", align_corners=False
            )
        return mem_feat

    def _flatten_memory_tokens(
        self,
        memory_feats: torch.Tensor,
        memory_pos: torch.Tensor,
        num_video_frames: int,
        device: torch.device,
    ):
        feats = memory_feats.to(device, non_blocking=True)
        pos = memory_pos.to(device, non_blocking=True)
        if feats.dim() == 4:
            feats = feats.unsqueeze(1)
        if pos.dim() == 4:
            pos = pos.unsqueeze(1)
        B, num_memories, C, H_mem, W_mem = feats.shape
        tokens_per_memory = H_mem * W_mem

        feats_flat = (
            feats.view(B, num_memories, C, tokens_per_memory)
            .permute(3, 0, 1, 2)
            .reshape(tokens_per_memory * num_memories, B, C)
        )
        pos_flat = (
            pos.view(B, num_memories, C, tokens_per_memory)
            .permute(3, 0, 1, 2)
            .reshape(tokens_per_memory * num_memories, B, C)
        )

        if num_video_frames > 1:
            feats_flat = feats_flat.repeat(1, num_video_frames, 1)
            pos_flat = pos_flat.repeat(1, num_video_frames, 1)

        if self.maskmem_tpos_enc is not None:
            slot_ids = torch.arange(num_memories, device=device).repeat_interleave(
                tokens_per_memory
            )
            slot_enc = self.maskmem_tpos_enc[
                (slot_ids % self.maskmem_tpos_enc.size(0)).clamp(
                    max=self.maskmem_tpos_enc.size(0) - 1
                )
            ]
            slot_enc = slot_enc.view(-1, self.mem_dim).unsqueeze(1)  # [TN, 1, C]
            slot_enc = slot_enc.expand(-1, B, -1)
            if num_video_frames > 1:
                slot_enc = slot_enc.repeat(1, num_video_frames, 1)
            pos_flat = pos_flat + slot_enc

        return feats_flat, pos_flat
