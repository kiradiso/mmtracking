import math
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from ..utils import gaussian_radius, draw_umich_gaussian

@PIPELINES.register_module()
class CTrackTarget(object):
    """Matching objects on a pair of images.

    Args:
        opt (dict): Config from CenterTrack.
    """

    def __init__(self, opt):
        # opt -> hm_disturb, lost_disturb
        self.opt = opt

    def _get_pre_dets(self, result):
        """
        CTrack random sample nearby frame as the "previoud" frame in training.
        Due to only sample one frame, it can follow ref_sampler in CocoVideoDataset 
        for both training and testing.
        Code change from CenterTrack.
        // PS: Ignore, area < 0, category_id not in dataset.cat_ids have be filtered in _parse_ann_info.
        // PS: In CornerNet .etc, target is generated in dense_head.

        Args:
            result (dict): Contain ref image infos.

        Returns:
            tuple: pre_hm, pre_cts, track_ids, filtered by boxh == 0 or boxw == 0.
        """
        hm_h, hm_w, _ = result['pad_shape']
        reutrn_hm = self.opt['reutrn_hm']
        down_ratio = self.opt['down_ratio']
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, track_ids = [], []
        for idx, bbox in enumerate(result['gt_bboxes']): # ndarray, nx4, xyxy
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius)) 
                max_rad = max(max_rad, radius)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct0 = ct.copy()
                conf = 1

                ct[0] = ct[0] + np.random.randn() * self.opt['hm_disturb'] * w
                ct[1] = ct[1] + np.random.randn() * self.opt['hm_disturb'] * h
                conf = 1 if np.random.random() > self.opt['lost_disturb'] else 0
                
                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct)
                else:
                    pre_cts.append(ct0)
                track_ids.append(result['gt_instance_ids'][idx])

                if reutrn_hm:
                    draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                if np.random.random() < self.opt['fp_disturb'] and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h 
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

        return pre_hm, pre_cts, track_ids

    def __call__(self, results):
        if len(results) != 2:
            print("Get result length : ", len(results))
            raise NotImplementedError('Only support match 2 images now.')

        pre_hm, pre_cts, track_ids = self._get_pre_dets(results[1])
        # TODO: disturb_ref_ids is not collectable
        # If collect, it should be added to SeqDefaultFormatBundle
        results[1]['disturb_ref_ids'] = track_ids
        results[0]['pre_hm'] = pre_hm
        results[0]['pre_cts'] = pre_cts
        results[1]['pre_hm'] = np.zeros_like(pre_hm)
        results[1]['pre_cts'] = []
        # Process like SeqDefaultFormatBundle
        # If result is Tensor, it will stack, shape must be consistent
        # If result is list, it will call zip(*batch), batch for each index
        # len should be same, otherwise will be dropped
        # If result is DC, it will be treat as Tensor, but support
        # inconsistent shape.
        for _results in results:
            for key in ['pre_cts']:
                if key not in _results:
                    continue
                _results[key] = DC(to_tensor(_results[key]))

        return results
