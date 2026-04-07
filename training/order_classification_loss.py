# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional, Tuple, Dict
import torch.distributed as dist
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)
from training.utils import get_non_diag_values, cannot_compare_instances


class OrderClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        occlusion_coefficient: float,
        depth_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.occlusion_coefficient = occlusion_coefficient
        self.depth_coefficient = depth_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: Tensor,
        targets: List[dict],
        class_queries_logits: Optional[Tensor] = None,
        occlusion_logits: Optional[List[Tensor]] = None,
        depth_logits: Optional[List[Tensor]] = None,
    ):
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]
        occlusion_labels = [target["occlusion"].long() for target in targets]
        depth_labels = [target["depth"].long() for target in targets]

        indices: List[Tuple[Tensor, Tensor]] = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)
        loss = {**loss_masks, **loss_classes}
        # InstaOrder
        if occlusion_logits is not None:
            loss_occ = self.loss_occlusion(occlusion_logits, occlusion_labels, indices)
            loss = {**loss, **loss_occ}
        if depth_logits is not None:
            loss_depth = self.loss_depth(depth_logits, depth_labels, indices)
            loss = {**loss, **loss_depth}

        return loss

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_occlusion(
        self,
        outputs: List[Tensor],
        targets: List[Tensor],
        indices: List[Tensor],
    ) -> Dict[str, Tensor]:
        loss = outputs[0].new_tensor(0.0)
        for src_occ, tgt_occ, match in zip(outputs, targets, indices):
            # Skip samples for which only one instance has been detected
            # since there are no occlusions to compute
            if cannot_compare_instances(src_occ):
                continue

            src_idx, tgt_idx = match
            # Re-ordering predictions
            src_occ = src_occ[src_idx, :][:, src_idx]
            # Re-index the targets to match the predictions
            tgt_occ = tgt_occ[tgt_idx, :][:, tgt_idx]

            # IMPORTANT: Checking that the matrix has still some valid
            # samples. This is important at the beginning of the training
            # since the masks are not predicted well, there is often just
            # 1 detected mask. Loss then compares [] and [], leading to NaNs.
            if cannot_compare_instances(src_occ):
                continue

            # Only penalize the network for non-diagonal values
            # (i.e. occlusion order between different instances)
            tgt_occ = get_non_diag_values(tgt_occ)
            src_occ = get_non_diag_values(src_occ)

            tgt_occ = F.one_hot(tgt_occ, num_classes=2)
            loss += F.binary_cross_entropy_with_logits(src_occ, tgt_occ.float())
        return {"occlusion": loss}

    def loss_depth(
        self,
        outputs: List[Tensor],
        targets: List[Tensor],
        indices: List[Tensor],
    ) -> Dict[str, Tensor]:

        loss = outputs[0].new_tensor(0.0)
        for src_depth, tgt_depth, match in zip(outputs, targets, indices):
            # Skip samples for which only one instance has been detected
            # since there are no occlusions to compute
            if cannot_compare_instances(src_depth):
                continue

            src_idx, tgt_idx = match
            # Re-ordering predictions
            src_depth = src_depth[src_idx, :][:, src_idx]
            # Re-index the targets to match the predictions
            tgt_depth = tgt_depth[tgt_idx, :][:, tgt_idx]

            # IMPORTANT: Checking that the matrix has still some valid
            # samples. This is important at the beginning of the training
            # since the masks are not predicted well, there is often just
            # 1 detected mask. Loss then compares [] and [], leading to NaNs.
            if cannot_compare_instances(src_depth):
                continue

            # Only penalize the network for non-diagonal values
            # (i.e. depth order between different instances)
            tgt_depth = get_non_diag_values(tgt_depth)
            src_depth = get_non_diag_values(src_depth)

            tgt_depth = F.one_hot(tgt_depth, num_classes=3).float()
            loss += F.cross_entropy(src_depth, tgt_depth)
        return {"loss_depth": loss}

    def loss_total(self, losses_all_layers, log_fn) -> Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            elif "occlusion" in loss_key:
                weighted_loss = loss * self.occlusion_coefficient
                print(f"Occ. loss {weighted_loss:.3f} (w/o weight {loss:.3f})")
            elif "depth" in loss_key:
                weighted_loss = loss * self.depth_coefficient
                print(f"Depth loss {weighted_loss:.3f} (w/o weight {loss:.3f})")
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
