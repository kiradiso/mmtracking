from mmdet.core import bbox2result
from mmdet.models.builder import build_backbone, build_head

from mmtrack.core import track2result
from ..builder import (MODELS, build_detector, build_motion, build_reid,
                       build_tracker)
from ..motion import CameraMotionCompensation, LinearMotion
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class CenterTrack(BaseMultiObjectTracker):
    """Tracking Object as Points.

    From official github, backbone corr backbone + head in mmdet,
    head only for forward loss.
    """

    def __init__(self,
                 backbone=None,
                 head=None,
                 tracker=None,
                 pretrains=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if head is not None:
            head = head.copy()
            head.update(train_cfg=train_cfg.dense_head, test_cfg=test_cfg.dense_head)
            self.track_head = build_head(head)

        if tracker is not None:
            tracker = tracker.copy()
            tracker.update(test_cfg=test_cfg.tracker)
            self.tracker = build_tracker(tracker)

        self.init_weights(pretrains)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.fp16_enabled = True

    def init_weights(self, pretrain):
        """Initialize the weights of the modules.

        Args:
            pretrained (dict): Path to pre-trained weights.
        """
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if pretrain.get('backbone', False):
            ckpt, load_type = pretrain['backbone']
            if load_type == 'mmcv':
                self.init_module('backbone', ckpt)
            else:
                self.backbone.init_weights(ckpt)

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
            output['hm'], output['reg'], output['tracking'], output['ltrb_amodal'], 
            gt_bboxes, gt_labels, pre_cts, gt_match_indices, img_metas,
            gt_bboxes_amodal=gt_bboxes_amodal)
        return loss

    def simple_test(self,
                    img,
                    img_metas,
                    rescale=False,
                    public_bboxes=None,
                    **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)     # In fact, here assume N=1
        # print('===', frame_id, "Frame_id")
        if frame_id == 0:
            print("Reset track!!!")
            self.tracker.reset()        # frame_id == 0 is the key to reset!!
            self.pre_img = None

        if self.pre_img is None:
            self.pre_img = img
            # TODO: init tracker like CenterTrack, can use public detection with pre_det/cur_det

        # render input heatmap from tracker status
        # pre_inds is not used in the current version.
        # We used pre_inds for learning an offset from previous image to
        # the current image.
        # The buffer bboxes are rescale by get_bboxes, so here need to be consistent.
        pre_hm, pre_inds = self.tracker.prepare_heatmap(
            frame_id, img_metas, img.device, rescale=rescale)

        x = self.backbone(img, pre_img=self.pre_img, pre_hm=pre_hm)
        # Process detection results, list for different images
        result_list = self.track_head.process(x, img_metas, pre_inds, rescale=rescale)

        # Merge output, due to simple_test, it take no effect
        # pass

        assert public_bboxes is None, "Not support public detections now!"
        

        num_classes = self.track_head.num_classes

        bboxes, labels, ids = self.tracker.track(
            dets=result_list[0]['detections'],
            det_cts=result_list[0]['cts'],
            det_cats=result_list[0]['clses'],
            tracking_offs=result_list[0]['tracking'],
            frame_id=frame_id,
            img_metas=img_metas,
            public_det=public_bboxes,
            **kwargs)
        self.pre_img = img

        track_result = track2result(bboxes, labels, ids, num_classes)
        bbox_result = bbox2result(
            result_list[0]['detections'], result_list[0]['clses'], num_classes)
        
        debug = False
        k = 0
        if debug:
            img_name = img_metas[0]['ori_filename'].split('/')[-1]
            # ori_full_path = 'data/MOT17/' + 'train/' + img_metas[0]['ori_filename']
            ori_full_path = img_metas[0]['ori_filename']
            self.show_result(ori_full_path, track_result, out_file='data/experiments/CenterTrack/debug/{}'.format(img_name))
            k+=1
            if k > 90:
                _ = input()
        return dict(bbox_results=bbox_result, track_results=track_result)