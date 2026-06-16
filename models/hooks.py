# Copyright (c) OpenMMLab. All rights reserved.
"""Custom training hooks for YOLO26 training.

This module provides hooks for the YOLO26 training pipeline, including:
- ProgressiveLossHook: schedules the O2O head loss weight during training
"""

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
