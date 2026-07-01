"""Build the pure-PyTorch OrientedRCNN detector and load trained weights.

No mmcv/mmdet/mmrotate. The module structure mirrors the trained checkpoint
exactly (backbone.adapter.*, neck.*, rpn_head.*, roi_head.*) so the non-EMA
state_dict keys (= EMA weights that produced the reported mAP) load directly.
"""

import torch

from .core.detector import build_detector_from_checkpoint
from .data.dior_data import CLASSES


def build_model(checkpoint, num_classes=20, device='cuda'):
    model = build_detector_from_checkpoint(checkpoint, num_classes=num_classes)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = ck.get('state_dict', ck)
    clean = {k: v for k, v in state_dict.items() if not k.startswith('ema_')}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    real_missing = [k for k in missing if 'num_batches_tracked' not in k]
    if real_missing:
        print('WARNING missing keys (first 20):', real_missing[:20])
    model = model.to(device)
    model.eval()
    classes = ck.get('meta', {}).get('CLASSES', CLASSES)
    return model, classes
