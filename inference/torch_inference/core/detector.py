"""Pure-PyTorch detectors: OrientedRCNN and RotatedFCOS (DINOv3 ViT-Adapter)."""

import torch
import torch.nn as nn

from . import box_ops
from .fcos_head import RotatedFCOSHead
from .fpn import FPN
from .roi import OrientedStandardRoIHead, RotatedShared2FCBBoxHead, RotatedSingleRoIExtractor
from .rpn import OrientedRPNHead
from .vit_adapter import DINOv3ViTAdapter


def _build_backbone_neck():
    backbone = DINOv3ViTAdapter(
        model_name='dinov3_vitl16', interaction_indexes=(5, 11, 17, 23),
        pretrain_size=512, conv_inplane=64, n_points=4, deform_num_heads=16,
        drop_path_rate=0.3, cffn_ratio=0.25, deform_ratio=0.5,
        use_extra_extractor=True)
    neck = FPN(in_channels=(1024,) * 4, out_channels=256, num_outs=5,
               start_level=0, add_extra_convs='on_output',
               relu_before_extra_convs=True)
    return backbone, neck


class OrientedRCNN(nn.Module):
    """Two-stage oriented detector: ViT-Adapter -> FPN -> OrientedRPN -> RoI."""

    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone, self.neck = _build_backbone_neck()
        self.rpn_head = OrientedRPNHead(in_channels=256, feat_channels=256,
                                        num_anchors=3, scales=(8.0,),
                                        ratios=(0.5, 1.0, 2.0), strides=(4, 8, 16, 32, 64))
        self.roi_head = OrientedStandardRoIHead(
            bbox_roi_extractor=RotatedSingleRoIExtractor(
                out_channels=256, featmap_strides=(4, 8, 16, 32), out_size=7, finest_scale=56),
            bbox_head=RotatedShared2FCBBoxHead(
                in_channels=256, fc_out_channels=1024, roi_feat_size=7,
                num_classes=num_classes, reg_class_agnostic=True))

    def extract_feat(self, img):
        return self.neck(self.backbone(img))

    @torch.no_grad()
    def simple_test(self, img, img_metas, rescale=True):
        """img: (1,C,H,W) tensor; img_metas: list[dict]. Returns per-image
        per-class detection lists (eval_rbbox_map format)."""
        feats = self.extract_feat(img)
        proposals = self.rpn_head.simple_test_rpn(feats, img_metas)
        return self.roi_head.simple_test(feats, proposals, img_metas, rescale=rescale)


class RotatedFCOS(nn.Module):
    """Anchor-free one-stage detector: ViT-Adapter -> FPN -> RotatedFCOSHead."""

    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone, self.neck = _build_backbone_neck()
        self.bbox_head = RotatedFCOSHead(
            num_classes=num_classes, in_channels=256, feat_channels=256,
            stacked_convs=4, strides=(4, 8, 16, 32, 64), norm_on_bbox=True,
            centerness_on_reg=True, scale_angle=True, num_groups=32)

    def extract_feat(self, img):
        return self.neck(self.backbone(img))

    @torch.no_grad()
    def simple_test(self, img, img_metas, rescale=True):
        feats = self.extract_feat(img)
        cls_scores, bbox_preds, angle_preds, centernesses = self.bbox_head(feats)
        results = []
        for i, meta in enumerate(img_metas):
            cs = [cls_scores[l][i] for l in range(len(cls_scores))]
            bp = [bbox_preds[l][i] for l in range(len(bbox_preds))]
            ap = [angle_preds[l][i] for l in range(len(angle_preds))]
            cn = [centernesses[l][i] for l in range(len(centernesses))]
            det, labels = self.bbox_head.get_bboxes_single(
                cs, bp, ap, cn, meta['img_shape'], meta['scale_factor'],
                nms_pre=2000, score_thr=0.05, nms_iou=0.1, max_per_img=2000,
                rescale=rescale, device=img.device)
            results.append(box_ops.rbbox2result(det, labels, self.bbox_head.num_classes))
        return results


def build_detector_from_checkpoint(checkpoint, num_classes=20):
    """Auto-detect detector type from the checkpoint state_dict keys."""
    import torch
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    keys = set(sd.get('state_dict', sd).keys())
    has = lambda s: any(k.startswith(s) for k in keys)
    if has('rpn_head.') and has('roi_head.'):
        return OrientedRCNN(num_classes=num_classes)
    if has('bbox_head.conv_centerness'):
        return RotatedFCOS(num_classes=num_classes)
    raise ValueError('Unknown detector type in checkpoint (no rpn_head/roi_head or '
                     'bbox_head.conv_centerness found).')
