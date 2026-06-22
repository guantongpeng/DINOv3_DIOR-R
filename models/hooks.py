# Copyright (c) OpenMMLab. All rights reserved.
"""Custom training hooks for YOLO26 training.

This module provides hooks for the YOLO26 training pipeline, including:
- ProgressiveLossHook: schedules the O2O head loss weight during training
"""

import torch.nn as nn
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ProgressiveLossHook(Hook):
    """Hook to manage the progressive O2O (one-to-one) loss weight schedule.

    During YOLO26 training, the loss should gradually shift from the O2M
    (one-to-many) head to the O2O (one-to-one) head. This hook updates
    the O2O weight on the bbox_head before each epoch.

    Schedule:
        - Before start_epoch: o2o_weight = 0.0 (only O2M head)
        - Between start_epoch and end_epoch: o2o_weight linearly increases from 0 to 1
        - After end_epoch: o2o_weight = 1.0 (equal O2M and O2O weight)

    Args:
        start_epoch (int): Epoch at which to start increasing O2O weight.
        end_epoch (int): Epoch at which O2O weight reaches 1.0.
    """

    def __init__(self, start_epoch: int = 12, end_epoch: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def before_train_epoch(self, runner):
        """Update O2O weight before each training epoch.

        Args:
            runner: The runner object containing the model.
        """
        epoch = runner.epoch
        model = runner.model

        # Compute progressive weight
        if epoch < self.start_epoch:
            weight = 0.0
        elif epoch >= self.end_epoch:
            weight = 1.0
        else:
            # Linear ramp from 0 to 1
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            weight = progress

        # Set O2O weight on the bbox_head
        if hasattr(model, 'module'):
            # Distributed training
            model = model.module

        if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'o2o_weight'):
            model.bbox_head.o2o_weight = weight
            runner.logger.info(
                f'[ProgressiveLoss] Epoch {epoch}: '
                f'O2O weight = {weight:.3f}'
            )


@HOOKS.register_module()
class RegZeroInitHook(Hook):
    """Zero-init the RoI bbox regression head (DETR-style stable init).

    PlainDETR (official DINOv3 detection eval) inits the box regression MLP so
    that the very first regression output is zero — i.e. the predicted box
    equals the anchor/proposal, which stabilizes early training. This hook
    applies the equivalent to an MMDetection RotatedShared2FCBBoxHead:
    the final regression FC (`fc_reg`) gets weight=0, bias=0.

    It runs once at training start (before the first epoch). If the head does
    not expose `fc_reg`, it logs a warning and does nothing.

    Args:
        zero_weight (bool): zero the regression FC weight. Default True.
        zero_bias (bool): zero the regression FC bias. Default True.
    """

    def __init__(self, zero_weight: bool = True, zero_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.zero_weight = zero_weight
        self.zero_bias = zero_bias

    def before_run(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        heads = []
        # Detect two-stage detector layout: roi_head.bbox_head (single or list).
        roi = getattr(model, 'roi_head', None)
        if roi is not None:
            bh = getattr(roi, 'bbox_head', None)
            if bh is None:
                return
            heads = bh if isinstance(bh, (list, tuple, nn.ModuleList)) else [bh]

        applied = 0
        for head in heads:
            fc_reg = getattr(head, 'fc_reg', None)
            if fc_reg is None:
                runner.logger.warning(
                    '[RegZeroInit] bbox_head has no fc_reg; skipping zero-init.')
                continue
            if self.zero_weight:
                nn.init.constant_(fc_reg.weight.data, 0.0)
            if self.zero_bias and fc_reg.bias is not None:
                nn.init.constant_(fc_reg.bias.data, 0.0)
            applied += 1
        if applied:
            runner.logger.info(
                f'[RegZeroInit] zero-initialised regression FC in {applied} bbox_head(s).')
