"""Convert a RotatedFCOS (DINOv3 ViT-Adapter) .pth checkpoint to ONNX.

Only the network *forward* (backbone + neck + bbox_head conv tower) is exported.
The FCOS post-processing (point grid, sigmoid, distance2obb decode, rotated NMS)
is intentionally kept out of the graph -- it is done in numpy/torch afterwards
(``postprocess.py``). This is what makes the export tractable: the custom CUDA
ops (rotated NMS) and the heavy control-flow in get_bboxes stay in Python.

Three things that would otherwise break torch.onnx.export are handled here:
  1. ``MultiScaleDeformableAttention`` calls the mmcv CUDA op on GPU. We force
     the pure-PyTorch ``grid_sample`` path so the op is traceable and runs in
     onnxruntime (CPU *and* GPU).
  2. The ViT is exported in fp32 (the config's ``bf16_vit`` autocast is only
     applied on the frozen-ViT path, which is not taken with freeze_vit=False).
  3. Gradient checkpointing (``with_cp``) is a no-op under tracing (requires_grad
     is False in eval), so it is left untouched.

The input is fixed at (1, 3, 800, 800) -- every DIOR-R image is 800x800 and the
test pipeline pads to a multiple of 32, so 800x800 is the exact inference shape.
For arbitrary images the caller resizes+letterbox-pads to 800x800.

Usage:
    python inference/onnx_inference/export/convert.py \\
        --config work_dirs/.../stage2/rotated_fcos_..._trainval_dior.py \\
        --checkpoint work_dirs/.../stage2/best_mAP@0.50_epoch_39.pth \\
        --out inference/onnx_inference/models/model.onnx
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn

PROJ = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
sys.path.insert(0, PROJ)
# dinov3 source tree (added to sys.path by the backbone at import time too; we
# need it earlier so the trace-patches can import the adapter helpers).
_DINOV3_SRC = osp.join(PROJ, 'third_party', 'dinov3')
if osp.isdir(_DINOV3_SRC) and _DINOV3_SRC not in sys.path:
    sys.path.insert(0, _DINOV3_SRC)

from mmcv import Config  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

import mmcv.cnn.bricks.transformer as _mmcv_transformer  # noqa: E402

IMG_SIZE = 800


def patch_msda_to_pytorch():
    """Force MultiScaleDeformableAttention onto the traceable grid_sample path.

    mmcv's ``MultiScaleDeformableAttention`` (defined in
    ``mmcv.ops.multi_scale_deform_attn``) picks the custom CUDA autograd op
    ``MultiScaleDeformableAttnFunction`` whenever the tensor is on CUDA. That
    op is opaque to ``torch.onnx``: the tracer cannot look inside it, so it
    captures the whole attention sub-graph (incl. its Linear projections) as a
    single *constant* -- the output then never responds to the input, which
    wrecks accuracy.

    We flip the module-level ``IS_CUDA_AVAILABLE`` the forward reads to False
    so it uses the pure-PyTorch ``grid_sample`` implementation instead. That
    implementation feeds tensor-typed sizes into ``value.split``/``reshape``
    which the tracer also mishandles, so it is replaced with an equivalent
    that uses python ints via ``.tolist()`` (exact for fixed input sizes).
    """
    _mmcv_transformer.IS_CUDA_AVAILABLE = False
    import torch.nn.functional as _F
    import mmcv.ops.multi_scale_deform_attn as _msda_mod

    def _msda_pytorch(value, value_spatial_shapes, sampling_locations,
                      attention_weights):
        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
        shapes = value_spatial_shapes.tolist()
        value_list = value.split([int(H) * int(W) for H, W in shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (H, W) in enumerate(shapes):
            H, W = int(H), int(W)
            value_l = value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H, W)
            sampling_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            sampling_value_list.append(_F.grid_sample(
                value_l, sampling_grid, mode='bilinear',
                padding_mode='zeros', align_corners=False))
        attention_weights = attention_weights.transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_levels * num_points)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)
                  * attention_weights).sum(-1).view(
            bs, num_heads * embed_dims, num_queries)
        return output.transpose(1, 2).contiguous()

    # Patch the module the class is DEFINED in (its forward reads these globals).
    _msda_mod.IS_CUDA_AVAILABLE = False
    _msda_mod.multi_scale_deformable_attn_pytorch = _msda_pytorch
    # Some mmcv builds re-export the name into the transformer module too.
    _mmcv_transformer.IS_CUDA_AVAILABLE = False
    _mmcv_transformer.multi_scale_deformable_attn_pytorch = _msda_pytorch


def patch_deform_inputs_to_constants():
    """Replace ``torch.linspace``/``arange`` (-> ONNX ``Range`` op) in the
    DINOv3 adapter helpers with numpy-derived constant initializers.

    The reference points / spatial shapes / level start indices depend only on
    the (fixed) input size, so emitting them as ONNX constants is exact and
    avoids a torch.onnx device-inference crash on the ``Range`` symbolic
    ("tensor does not have a device").
    """
    import numpy as np
    import torch

    import dinov3.eval.segmentation.models.backbone.dinov3_adapter as da

    def get_reference_points(spatial_shapes, device):
        out = []
        for (H_, W_) in spatial_shapes:
            ys = (np.arange(H_) + 0.5).astype(np.float32) / H_
            xs = (np.arange(W_) + 0.5).astype(np.float32) / W_
            gx, gy = np.meshgrid(xs, ys)
            ref = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)[None]
            out.append(torch.from_numpy(ref))
        reference_points = torch.cat(out, dim=1)[:, :, None]
        return reference_points.to(device)

    def deform_inputs(x, patch_size):
        bs, c, h, w = x.shape
        shapes = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
        spatial_shapes = torch.tensor(shapes, dtype=torch.long, device=x.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        reference_points = get_reference_points(
            [(h // patch_size, w // patch_size)], x.device)
        deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

        spatial_shapes = torch.tensor(
            [(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        reference_points = get_reference_points(shapes, x.device)
        deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
        return deform_inputs1, deform_inputs2

    da.get_reference_points = get_reference_points
    da.deform_inputs = deform_inputs


def patch_rope_to_constants():
    """Patch the DINOv3 RoPE forward to emit its sin/cos as ONNX constants,
    *preserving the RoPE's native dtype (bf16)*.

    The original builds coords with ``torch.arange`` (-> ONNX ``Range`` op),
    which trips a torch.onnx device-inference crash, so for a fixed input size
    the coords are emitted as numpy-derived constants instead.

    CRITICAL -- dtype must be preserved (bf16 for this checkpoint):
    ``attention.SelfAttention.apply_rope`` casts ``q``/``k`` to ``sin.dtype`` and
    performs the rotary rotation in that precision (then casts back). The
    checkpoint was trained/eval'd with the RoPE buffer in bf16, so the rotation
    runs in bf16 natively. The previous version of this patch forced fp32
    (``self.periods.float()`` + fp32 coords), which silently rewrote every RoPE
    rotation to fp32 and drifted the backbone features by ~1-2% (rel). The
    single-stage heads tolerate that, but the two-stage OrientedRCNN RPN->RoI
    cascade amplifies it into a ~5-point mAP gap (0.6992 vs 0.7500). Keeping
    bf16 reproduces the reference exactly (max|diff| vs native = 0).

    onnxruntime does NOT accept bfloat16 as the input type of the ``Sin``/``Cos``
    operators, so we cannot let those ops reach the graph. Instead the sin/cos
    are *precomputed* in native bf16 (matching the reference bit-for-bit) and
    cached; the patched ``forward`` returns the cached value cast back to fp32
    (lossless: the bf16-rounded value, stored as fp32). A warm-up forward
    (``_export``) populates the cache *before* tracing, so the tracer records the
    cached value as an ONNX fp32 constant -- no ``Sin`` op is emitted.
    ``patch_apply_rope_bf16`` then forces the rotary rotation itself to run in
    bf16 (casting the fp32-stored sin/cos back to bf16), reproducing the native
    bf16 rotation exactly while keeping every ONNX op fp32-typed except the
    supported bf16 ``Mul``/``Add``/``Cast`` inside the rotation.
    """
    import math

    import numpy as np
    import torch

    from dinov3.layers.rope_position_encoding import RopePositionEmbedding

    _cache = {}  # (id(self), H, W) -> (sin, cos) fp32 tensors (bf16-rounded values)

    def forward(self, *, H: int, W: int):
        key = (id(self), int(H), int(W))
        cached = _cache.get(key)
        if cached is None:
            device = self.periods.device
            dt = self.periods.dtype  # native bf16 -- keep it (see docstring)
            H_, W_ = int(H), int(W)
            nc = self.normalize_coords
            if nc == 'separate':
                dH, dW = H_, W_
            elif nc == 'max':
                dH = dW = max(H_, W_)
            elif nc == 'min':
                dH = dW = min(H_, W_)
            else:
                raise ValueError(nc)
            # numpy fp32 coords (avoids ONNX Range), cast to the native bf16
            # dtype. The half-integer values are exactly representable in bf16,
            # so this equals torch.arange(0.5, H, dtype=bf16) / H.
            ch = torch.from_numpy(
                (np.arange(H_, dtype=np.float32) + 0.5) / float(dH)).to(device).to(dt)
            cw = torch.from_numpy(
                (np.arange(W_, dtype=np.float32) + 0.5) / float(dW)).to(device).to(dt)
            coords = torch.stack(torch.meshgrid(ch, cw, indexing='ij'), dim=-1)
            coords = coords.flatten(0, 1)
            coords = 2.0 * coords - 1.0
            periods = self.periods  # native bf16 (was: self.periods.float())
            angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
            angles = angles.flatten(1, 2).tile(2)
            # compute in bf16 (native), store the bf16-rounded values as fp32 so
            # the ONNX constant is fp32-typed (no bf16 Sin op). apply_rope casts
            # them back to bf16 for the rotation (see patch_apply_rope_bf16).
            cached = (torch.sin(angles).float(), torch.cos(angles).float())
            _cache[key] = cached
        return cached

    RopePositionEmbedding.forward = forward


def patch_apply_rope_bf16():
    """Replicate the native bf16 rotary rotation in pure fp32 (ONNX-safe).

    ``patch_rope_to_constants`` returns the sin/cos as fp32 tensors (so the ONNX
    graph has no bf16 ``Sin``/``Cos``). The native ``apply_rope`` rotates in
    ``sin.dtype``, which would now be fp32 -- silently flipping the rotation to
    fp32 and re-introducing the ~1-2% feature drift.

    This patch reproduces the native bf16 rotation *exactly* but with only fp32
    arithmetic plus ``Cast`` round-trips through bf16: PyTorch's bf16 elementwise
    ops compute in fp32 and round the result to bf16, so rounding each
    intermediate (``x``, each product, the sum) to bf16 precision via
    ``x.to(bfloat16).float()`` matches the native bf16 ``Mul``/``Add`` bit-for-bit
    (verified: max|diff| vs native = 0). Crucially the ONNX graph then contains
    only fp32 ``Mul``/``Add`` and bf16<->fp32 ``Cast`` -- no bf16 ``Mul`` -- so it
    is deterministic in onnxruntime (which otherwise diverges from PyTorch on
    bf16 elementwise ops) and reproduces the reference exactly at inference.
    """
    import torch

    import dinov3.layers.attention as _att
    from dinov3.layers.attention import rope_rotate_half

    def _r16(x):
        # round-to-bf16-precision, fp32 storage (== PyTorch bf16 elementwise math)
        return x.to(torch.bfloat16).float()

    def _rope_apply(x, sin, cos):
        x = _r16(x)
        return _r16(_r16(x * cos) + _r16(rope_rotate_half(x) * sin))

    def apply_rope(self, q, k, rope):
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        sin = _r16(sin); cos = _r16(cos)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        q_prefix = q[:, :, :prefix, :]
        q = _rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = _rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        return q.to(q_dtype), k.to(k_dtype)

    _att.SelfAttention.apply_rope = apply_rope


def load_model(config, checkpoint, device='cuda'):
    cfg = Config.fromfile(config)
    model = build_detector(cfg.model)
    model.eval()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = ck.get('state_dict', ck)
    # Use the EMA weights -- they are what produced the reported mAP. The EMA
    # keys are prefixed with `ema.`; the non-EMA `state_dict` keys mirror them.
    # Both load into the bare module keys; the non-EMA state_dict is what the
    # mmrotate evaluator uses, so we keep it (strip any residual ema_ keys).
    clean = {k: v for k, v in state_dict.items() if not k.startswith('ema_')}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    real_missing = [k for k in missing if 'num_batches_tracked' not in k]
    if real_missing:
        print('WARNING missing keys (first 20):', real_missing[:20])
    if unexpected:
        print('WARNING unexpected keys (first 20):', unexpected[:20])
    return model.to(device).eval()


class SingleStageWrapper(nn.Module):
    """backbone -> neck -> bbox_head. Outputs 5 levels x {cls, bbox, angle, X}.

    X is centerness (FCOS) or objectness (YOLO26). Each output is (1, C, H, W).
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.bbox_head = model.bbox_head

    def forward(self, img):
        feats = self.neck(self.backbone(img))
        cls_scores, bbox_preds, angle_preds, quals = self.bbox_head(feats)
        outs = []
        for lvl in range(len(cls_scores)):
            outs.append(cls_scores[lvl])
            outs.append(bbox_preds[lvl])
            outs.append(angle_preds[lvl])
            outs.append(quals[lvl])
        return tuple(outs)

    @staticmethod
    def output_names(fourth):
        names = []
        for lvl in range(5):
            for typ in ('cls', 'bbox', 'angle', fourth):
                names.append(f'f{lvl}_{typ}')
        return names


class ORCNNStageA(nn.Module):
    """OrientedRCNN stage A: backbone -> FPN -> RPN heads.

    Outputs: 4 FPN feature maps (strides 4/8/16/32, used by the RoI extractor)
    + 5 RPN cls maps + 5 RPN reg maps. RoI extraction + the bbox head live in
    stage B (and the RPN/RoI decode + NMS stays in Python).
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.rpn_head = model.rpn_head

    def forward(self, img):
        feats = self.neck(self.backbone(img))
        cls_scores, bbox_preds = self.rpn_head(feats)
        outs = [feats[l] for l in range(4)]
        outs += list(cls_scores) + list(bbox_preds)
        return tuple(outs)

    @staticmethod
    def output_names():
        return ([f'fpn{l}' for l in range(4)]
                + [f'rpn_cls{l}' for l in range(5)]
                + [f'rpn_reg{l}' for l in range(5)])


class ORCNNStageB(nn.Module):
    """OrientedRCNN stage B: RoI bbox head (2x FC -> cls + reg)."""

    def __init__(self, model):
        super().__init__()
        self.bbox_head = model.roi_head.bbox_head

    def forward(self, roi_feats):
        return self.bbox_head(roi_feats)  # (cls_score, bbox_pred)


def detect_detector(cfg):
    t = cfg.model['type']
    table = {'RotatedFCOS': 'fcos', 'DINOv3YOLO26': 'yolo26',
             'OrientedRCNN': 'oriented_rcnn'}
    if t not in table:
        raise ValueError(f'unsupported detector type: {t}')
    return table[t]


def build_meta(cfg, detector, paths):
    if detector == 'oriented_rcnn':
        nc = cfg.model.roi_head.bbox_head['num_classes']
    else:
        nc = cfg.model.bbox_head['num_classes']
    strides = list(cfg.model.bbox_head.get('strides', [4, 8, 16, 32, 64])) \
        if detector != 'oriented_rcnn' else \
        list(cfg.model.rpn_head.anchor_generator.get('strides', [4, 8, 16, 32, 64]))
    tc = cfg.model.get('test_cfg', {})
    meta = dict(detector=detector, num_classes=int(nc),
                strides=strides, img_size=IMG_SIZE, files=paths)
    if detector in ('fcos', 'yolo26'):
        meta['fourth_key'] = 'ctr' if detector == 'fcos' else 'obj'
        meta['test_cfg'] = dict(
            score_thr=tc.get('score_thr', 0.05),
            nms_iou_thr=tc.get('nms', {}).get('iou_thr', 0.1),
            max_per_img=tc.get('max_per_img', 2000),
            nms_pre=tc.get('nms_pre', 2000))
    else:  # oriented_rcnn
        rpn = tc.get('rpn', {})
        rcnn = tc.get('rcnn', {})
        meta['test_cfg'] = dict(
            rpn=dict(nms_pre=rpn.get('nms_pre', 2000),
                     max_per_img=rpn.get('max_per_img', 2000),
                     nms_iou=rpn.get('nms', {}).get('iou_threshold', 0.8)),
            rcnn=dict(nms_pre=rcnn.get('nms_pre', 2000),
                      score_thr=rcnn.get('score_thr', 0.05),
                      nms_iou=rcnn.get('nms', {}).get('iou_thr', 0.1),
                      max_per_img=rcnn.get('max_per_img', 2000)))
    return meta


def consolidate_onnx(path):
    """Merge the externalized weight files torch.onnx writes next to a large
    model back into a single self-contained ``.onnx`` (the ONNX protobuf limit
    is 2 GB; this model is ~1.4 GB so it fits inline) and delete the stray
    external files.

    Without this, torch.onnx.export (>=2.x) drops one file per initializer
    next to the model, which is messy and easy to break when moving the model.
    """
    import glob
    import onnx
    out_dir = osp.dirname(osp.abspath(path)) or '.'
    model = onnx.load(path, load_external_data=True)
    onnx.save_model(model, path, save_as_external_data=False)
    keep_ext = {'.onnx', '.py', '.sh', '.md', '.json'}
    for f in glob.glob(osp.join(out_dir, '*')):
        if osp.isfile(f) and osp.splitext(f)[1] not in keep_ext:
            os.remove(f)


def _export(wrapper, dummy, out, out_names, opset, no_fold):
    with torch.no_grad():
        # Warm-up forward: populates the RoPE sin/cos cache (patch_rope_to_constants)
        # so the trace records them as ONNX constants instead of emitting bf16
        # Sin/Cos ops (which onnxruntime rejects).
        wrapper(dummy)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), out, input_names=['input'],
            output_names=out_names, opset_version=opset,
            do_constant_folding=not no_fold, dynamic_axes=None)
    consolidate_onnx(out)


def main():
    import json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', default='inference/onnx_inference/models/model.onnx')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--verify', type=int, default=1)
    ap.add_argument('--no-constant-folding', action='store_true')
    args = ap.parse_args()

    patch_msda_to_pytorch()
    patch_deform_inputs_to_constants()
    patch_rope_to_constants()
    patch_apply_rope_bf16()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'[*] building model from {args.config}')
    cfg = Config.fromfile(args.config)
    detector = detect_detector(cfg)
    print(f'[*] detector = {detector}')
    model = load_model(args.config, args.checkpoint, device=device)
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)

    if detector in ('fcos', 'yolo26'):
        fourth = 'ctr' if detector == 'fcos' else 'obj'
        wrapper = SingleStageWrapper(model).to(device).eval()
        out_names = SingleStageWrapper.output_names(fourth)
        print(f'[*] exporting single-stage ONNX -> {args.out}')
        _export(wrapper, dummy, args.out, out_names, args.opset, args.no_constant_folding)
        meta = build_meta(cfg, detector, {'onnx': osp.basename(args.out)})
        if args.verify:
            verify(args, device, wrapper, dummy, out_names)
    else:  # oriented_rcnn
        stage_a = ORCNNStageA(model).to(device).eval()
        names_a = ORCNNStageA.output_names()
        out_b = osp.splitext(args.out)[0] + '_head.onnx'
        print(f'[*] exporting stage A (backbone+fpn+rpn) -> {args.out}')
        _export(stage_a, dummy, args.out, names_a, args.opset, args.no_constant_folding)
        # stage B: bbox head, dynamic #rois
        stage_b = ORCNNStageB(model).to(device).eval()
        roi_dummy = torch.randn(4, 256, 7, 7, device=device)
        names_b = ['cls_score', 'bbox_pred']
        print(f'[*] exporting stage B (bbox head) -> {out_b}')
        with torch.no_grad():
            torch.onnx.export(
                stage_b, (roi_dummy,), out_b, input_names=['roi_feats'],
                output_names=names_b, opset_version=args.opset,
                do_constant_folding=not args.no_constant_folding,
                dynamic_axes={'roi_feats': {0: 'num_rois'},
                              'cls_score': {0: 'num_rois'},
                              'bbox_pred': {0: 'num_rois'}})
        meta = build_meta(cfg, detector, {'stage_a': osp.basename(args.out),
                                          'stage_b': osp.basename(out_b)})
        if args.verify:
            verify_orcnn(args, device, stage_a, stage_b, dummy, roi_dummy,
                         names_a, names_b)

    meta_path = osp.splitext(args.out)[0] + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[+] wrote meta -> {meta_path}')


def verify(args, device, wrapper, dummy, out_names):
    import onnx
    import onnxruntime as ort
    print('[*] checking ONNX IR validity')
    onnx.checker.check_model(onnx.load(args.out))
    print('[*] comparing torch vs onnxruntime outputs')
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if device == 'cuda' else ['CPUExecutionProvider'])
    sess = ort.InferenceSession(args.out, providers=providers)
    with torch.no_grad():
        torch_outs = wrapper(dummy)
    torch_np = [o.detach().float().cpu().numpy() for o in torch_outs]
    ort_outs = sess.run(out_names, {'input': dummy.detach().cpu().numpy()})
    max_diff = 0.0
    for name, a, b in zip(out_names, torch_np, ort_outs):
        d = float(np.max(np.abs(a - b)))
        max_diff = max(max_diff, d)
        print(f'    {name:<10s} shape={tuple(a.shape)} max|diff|={d:.3e}')
    print(f'[+] overall max|diff| (torch vs ort) = {max_diff:.3e}')
    print(f'[+] numerical check: {"PASS" if max_diff < 1.0 else "CHECK"}')


def verify_orcnn(args, device, stage_a, stage_b, dummy, roi_dummy, names_a, names_b):
    import onnx
    import onnxruntime as ort
    for p in (args.out, osp.splitext(args.out)[0] + '_head.onnx'):
        onnx.checker.check_model(onnx.load(p))
    print('[*] comparing torch vs onnxruntime (stage A + stage B)')
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if device == 'cuda' else ['CPUExecutionProvider'])
    sa = ort.InferenceSession(args.out, providers=providers)
    sb = ort.InferenceSession(osp.splitext(args.out)[0] + '_head.onnx', providers=providers)
    with torch.no_grad():
        a_t = stage_a(dummy)
        b_t = stage_b(roi_dummy)
    a_o = sa.run(names_a, {'input': dummy.detach().cpu().numpy()})
    b_o = sb.run(names_b, {'roi_feats': roi_dummy.detach().cpu().numpy()})
    worst = 0.0
    for nm, t, o in zip(names_a, [x.detach().float().cpu().numpy() for x in a_t], a_o):
        d = float(np.max(np.abs(t - o))); worst = max(worst, d)
        print(f'    {nm:<10s} max|diff|={d:.3e}')
    for nm, t, o in zip(names_b, [x.detach().float().cpu().numpy() for x in b_t], b_o):
        d = float(np.max(np.abs(t - o))); worst = max(worst, d)
        print(f'    {nm:<10s} max|diff|={d:.3e}')
    print(f'[+] overall max|diff| = {worst:.3e} -> {"PASS" if worst < 1.0 else "CHECK"}')


if __name__ == '__main__':
    main()
