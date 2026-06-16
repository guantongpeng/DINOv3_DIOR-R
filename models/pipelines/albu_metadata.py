"""Albu wrapper that handles metadata keys compatible with newer albumentations.

The default mmdet Albu transform passes ALL keys from the pipeline's ``results``
dict through to ``Compose(**results)``.  Newer albumentations versions reject
unknown keyword arguments.  This wrapper filters out metadata keys that are
not relevant to albumentations before calling the augmentation pipeline, then
restores them afterwards.
"""

import numpy as np
from mmdet.datasets.pipelines.transforms import Albu as _Albu
from mmdet.datasets.builder import PIPELINES

try:
    import albumentations
except ImportError:
    albumentations = None


# Keys that are relevant to albumentations (image, masks, bboxes, keypoints)
# plus label fields that need to be tracked through bbox transforms.
_ALBU_KNOWN_KEYS = frozenset({
    'image',          # main image (mapped from 'img' by keymap)
    'masks',          # segmentation masks
    'bboxes',         # bounding boxes (mapped from 'gt_bboxes' by keymap)
    'keypoints',      # keypoints
})


@PIPELINES.register_module()
class AlbuMetadata(_Albu):
    """Albu augmentation that handles metadata keys.

    Behaves identically to mmdet's ``Albu`` transform except that it
    strips metadata keys from the dict before passing it through to
    the underlying albumentations ``Compose``, and re-attaches them
    afterwards.  This prevents ``ValueError: Key xxx is not in available keys.``
    with newer albumentations releases.
    """

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        if 'masks' in results:
            from mmdet.core.mask.structures import PolygonMasks
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            if albumentations.__version__ < '0.5':
                results['masks'] = results['masks'].masks
            else:
                results['masks'] = [mask for mask in results['masks'].masks]

        # ---- Filter metadata keys before passing to Compose ----
        aug_keys = _ALBU_KNOWN_KEYS | frozenset(
            getattr(self.aug, '_available_keys', [])
        )
        metadata = {}
        aug_input = {}
        for k, v in results.items():
            if k in aug_keys:
                aug_input[k] = v
            else:
                metadata[k] = v

        results = self.aug(**aug_input)

        # Restore metadata keys
        results.update(metadata)

        # ---- Rest of original __call__ logic ----
        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            if self.filter_lost_elements:
                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], results['image'].shape[0],
                        results['image'].shape[1])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results
