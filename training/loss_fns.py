# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            # Handle two-stage outputs
            if isinstance(outs, dict) and "stage1" in outs and "stage2" in outs:
                # Stage 1 loss
                losses_s1 = self._forward(outs["stage1"], targets, num_objects)
                
                # Stage 2 loss
                losses_s2 = self._forward(outs["stage2"], targets, num_objects)
                
                # Combine losses (weighted sum, e.g., 0.5 * s1 + 1.0 * s2)
                for k, v in losses_s1.items():
                    losses[k] += v * 0.5 # Give Stage 1 less weight as it's the "initial guess"
                    
                for k, v in losses_s2.items():
                    losses[k] += v * 1.0 # Give Stage 2 full weight as it's the "refined prediction"
            else:
                # Standard single-stage output
                cur_losses = self._forward(outs, targets, num_objects)
                for k, v in cur_losses.items():
                    losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "loss_fp": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
            # Get the best predicted mask for false positive loss
            best_mask = src_masks[batch_inds, best_loss_inds]  # [N, H, W]
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou
            # Get the best predicted mask for false positive loss
            best_mask = src_masks[:, 0]  # [N, H, W]

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # False positive loss: penalize predictions when GT is empty
        # target_obj = 0 means empty GT, target_obj = 1 means has GT
        # We want to penalize when model predicts non-empty mask but GT is empty
        no_target = 1.0 - target_obj  # 1.0 when GT is empty, 0.0 when GT has target
        
        # Get predicted mask probabilities (after sigmoid)
        pred_mask_sigmoid = best_mask.sigmoid()  # [N, H, W]
        
        # False positive: GT is empty (no_target=1) but prediction has non-zero values
        # Penalize the mean of predicted mask probabilities when GT is empty
        # This encourages the model to predict all zeros when there's no target
        # Using mean instead of sum to keep loss magnitude reasonable
        pred_mask_mean = pred_mask_sigmoid.flatten(1).mean(dim=-1, keepdim=True)  # [N, 1]
        loss_fp = pred_mask_mean * no_target  # Only apply when GT is empty
        
        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class
        losses["loss_fp"] += (loss_fp.sum() / num_objects)  # Normalize by num_objects like other losses

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
