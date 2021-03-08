from mmdet.core import bbox2result
from mmdet.models.builder import build_backbone, build_head
from mmcv.runner import auto_fp16

from mmtrack.core import track2result
from ..builder import (MODELS, build_detector, build_motion, build_reid,
                       build_tracker)
from ..motion import CameraMotionCompensation, LinearMotion
from .centertrack import CenterTrack


@MODELS.register_module()
class OneTrack(CenterTrack):
    """Tracking Object as Points.

    From official github, backbone corr backbone + head in mmdet,
    head only for forward loss.
    """

    # @auto_fp16(apply_to=('img', 'ref_img', 'pre_hm',))
    def forward_train(self, img, img_metas,
        gt_bboxes, gt_labels, ref_img=None, pre_hm=None, 
        pre_cts=None, gt_match_indices=None, gt_bboxes_amodal=None, **kwargs):
        """
        // PS: ref_img come from SeqDefaultFormatBundle, which reference data's keys add ref as prefix.
        // PS: reference data is collect to one other dict by collect operation(e.g. concat).
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Not support model_out_list
        output = self.backbone(img, pre_img=ref_img, pre_hm=pre_hm)
        # From list(dict) to dict(list)
        output = dict(zip(output[0].keys(), zip(*[o.values() for o in output])))
        loss = self.track_head.loss(
            output['hm'], output['tracking'], output['ltrb_amodal'], 
            gt_bboxes, gt_labels, pre_cts, gt_match_indices, img_metas,
            gt_bboxes_amodal=gt_bboxes_amodal)
        return loss