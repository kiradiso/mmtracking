from math import ceil, log

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from mmcv.ops import batched_nms

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from ..mot.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead



@HEADS.register_module()
class CenterTrackHead(BaseDenseHead):
    """Head of CornerNet: Detecting Objects as Paired Keypoints.

    Code is modified from the `official github repo
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/
    kp.py#L73>`_ .

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_ .

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_feat_levels (int): Levels of feature from the previous module. 2
            for HourglassNet-104 and 1 for HourglassNet-52. Because
            HourglassNet-104 outputs the final feature and intermediate
            supervision feature and HourglassNet-52 only outputs the final
            feature. Default: 2.
        train_cfg (dict | None): Training config. Useless in CornerHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CornerHead. Default: None.
        loss_heatmap (dict | None): Config of corner heatmap loss. Default:
            GaussianFocalLoss.
        loss_embedding (dict | None): Config of corner embedding loss. Default:
            AssociativeEmbeddingLoss.
        loss_offset (dict | None): Config of corner offset loss. Default:
            SmoothL1Loss.
    """

    def __init__(self,
                 num_classes,
                 num_feat_levels=2,
                 train_cfg=None,
                 test_cfg=None,
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 loss_offset=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1)):
        super(CenterTrackHead, self).__init__()
        self.num_classes = num_classes
        self.corner_offset_channels = 2
        self.num_feat_levels = num_feat_levels
        self.loss_heatmap = build_loss(
            loss_heatmap) if loss_heatmap is not None else None
        self.loss_offset = build_loss(
            loss_offset) if loss_offset is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def forward(self, output):
        """Forward features from the upstream network.

        Args:
            output (list[dict]): Result list from backbone model e.g. dlaseg.

        Returns:
            tuple: Usually a tuple of corner heatmaps, offset heatmaps and
            embedding heatmaps.
                - tl_heats (list[Tensor]): Top-left corner heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - br_heats (list[Tensor]): Bottom-right corner heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - tl_embs (list[Tensor] | list[None]): Top-left embedding
                  heatmaps for all levels, each is a 4D-tensor or None.
                  If not None, the channels number is corner_emb_channels.
                - br_embs (list[Tensor] | list[None]): Bottom-right embedding
                  heatmaps for all levels, each is a 4D-tensor or None.
                  If not None, the channels number is corner_emb_channels.
                - tl_offs (list[Tensor]): Top-left offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  corner_offset_channels.
                - br_offs (list[Tensor]): Bottom-right offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  corner_offset_channels.
        """
        # multi_apply apply func to multiple inputs, return list corr to every input.
        # can add args, kw-args, e.g. feats, lvl_ind corr to x, lvl_ind in forward single
        pass

    def forward_single(self, x, lvl_ind, return_pool=False):
        pass

    def get_targets(self,
                    gt_bboxes,
                    gt_labels,
                    feat_shape,
                    img_shape,
                    pre_cts,
                    gt_match_indices,
                    gt_bboxes_amodal=None):
        """Generate centertrack targets.
        In dla34, it return list[dict](default), which contain all
        desired output(e.g. reg/tracking/ltrb_amodel...).
        So, this head only generate target for this output.
        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]

        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)
        # print("Check ratio in get_target", width_ratio, height_ratio)
        assert int(img_h/height) == int(img_w/width) == 4, "Not as desired"

        gt_ct_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_ct_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_ct_tracking = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        # ltrb_amodel
        gt_ct_ltrb_am = gt_bboxes[-1].new_zeros([batch_size, 4, height, width])

        for batch_id in range(batch_size):
            # Ground truth of corner embedding per image is a list of coord set
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]

                # Use coords in the feature level to generate ground truth
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio

                # Int coords on feature map/ground truth tensor
                center_x_idx = int(min(scale_center_x, width - 1))
                center_y_idx = int(min(scale_center_y, height - 1))

                # Generate gaussian heatmap
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius((scale_box_height, scale_box_width))
                radius = max(0, int(radius))
                gt_ct_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_ct_heatmap[batch_id, label], [center_x_idx, center_y_idx],
                    radius)

                # Generate center offset
                center_x_offset = scale_center_x - center_x_idx
                center_y_offset = scale_center_y - center_y_idx
                # # May overlapped by latter box with same int coord
                gt_ct_offset[batch_id, 0, center_y_idx, center_x_idx] = center_x_offset
                gt_ct_offset[batch_id, 1, center_y_idx, center_x_idx] = center_y_offset

                # Generate tracking offset
                # TODO: Solve pre_ct aug
                pre_id = gt_match_indices[batch_id][box_id]
                if pre_id != -1:
                    center_x_pre, center_y_pre = pre_cts[batch_id][pre_id]
                    center_x_pre = center_x_pre*width_ratio
                    center_y_pre = center_y_pre*height_ratio

                    # CenterTrack use pre center - int(cur center)
                    center_x_pre_offset = center_x_pre - center_x_idx
                    center_y_pre_offset = center_y_pre - center_y_idx
                    gt_ct_tracking[batch_id, 0, center_y_idx, center_x_idx] = center_x_pre_offset
                    gt_ct_tracking[batch_id, 1, center_y_idx, center_x_idx] = center_y_pre_offset

                # Generate amodel box ltrb
                if gt_bboxes_amodal is not None:
                    left_am, top_am, right_am, bottom_am = gt_bboxes_amodal[batch_id][box_id]

                    # Use coords in the feature level to generate ground truth
                    scale_left_am = left_am * width_ratio
                    scale_right_am = right_am * width_ratio
                    scale_top_am = top_am * height_ratio
                    scale_bottom_am = bottom_am * height_ratio

                    gt_ct_ltrb_am[batch_id, 0, center_y_idx, center_x_idx] = scale_left_am - center_x_idx
                    gt_ct_ltrb_am[batch_id, 1, center_y_idx, center_x_idx] = scale_top_am - center_y_idx
                    gt_ct_ltrb_am[batch_id, 2, center_y_idx, center_x_idx] = scale_right_am - center_x_idx
                    gt_ct_ltrb_am[batch_id, 3, center_y_idx, center_x_idx] = scale_bottom_am - center_y_idx
                else:
                    gt_ct_ltrb_am[batch_id, 0, center_y_idx, center_x_idx] = scale_left - center_x_idx
                    gt_ct_ltrb_am[batch_id, 1, center_y_idx, center_x_idx] = scale_top - center_y_idx
                    gt_ct_ltrb_am[batch_id, 2, center_y_idx, center_x_idx] = scale_right - center_x_idx
                    gt_ct_ltrb_am[batch_id, 3, center_y_idx, center_x_idx] = scale_bottom - center_y_idx

        target_result = dict(
            center_heatmap=gt_ct_heatmap,
            center_offset=gt_ct_offset,
            center_tracking_offset=gt_ct_tracking,
            center_ltrb=gt_ct_ltrb_am)

        return target_result

    def loss(self,
             ct_heats,
             ct_offs,
             tracking_offs,
             ct_ltrbs,
             gt_bboxes,
             gt_labels,
             pre_cts,
             gt_match_indices,
             img_metas,
             gt_bboxes_amodal=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            ct_heats (list[Tensor]): Center heatmaps for each level
                with shape (N, num_classes, H, W).
            ct_offs (list[Tensor]): Center offsets for each level
                with shape (N, corner_offset_channels, H, W).
            tracking_offs (list[Tensor]): Center Tracking offset.
            ct_ltrbs (list[Tensor]): Center ltrb.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. Containing the
            following losses:

                - det_loss (list[Tensor]): Corner keypoint losses of all
                  feature levels.
                - off_loss (list[Tensor]): Corner offset losses of all feature
                  levels.
                - tracking_loss
                - ltrb_losses
        """
        targets = self.get_targets(
            gt_bboxes,
            gt_labels,
            ct_heats[-1].shape,
            img_metas[0]['pad_shape'],
            pre_cts, gt_match_indices, gt_bboxes_amodal)
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        det_losses, off_losses, tracking_losses, ltrb_losses = multi_apply(
            self.loss_single, ct_heats, ct_offs, tracking_offs, ct_ltrbs, mlvl_targets)
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses, tracking_loss=tracking_losses, ltrb_losses=ltrb_losses)
        return loss_dict

    def loss_single(self, ct_hmp, ct_off, tracking_off, ct_ltrb, targets):
        """Compute losses for single level.
        Heatmap Focal Loss:
        Parameters: num_stacks: default to 1, 2 for hourglass(not support)
        weight: ltrb/wh/ltrb_amodel -> 0.1 others -> 1
        !!!Norm is sum of pos(N), use avg_factor to adjust.
        Regression Loss:
        L1/SmoothL1(in mmdet)
        off -> weight is 1
        tracking -> weight 1
        ltrb -> weight 0.1 

        Args:
            ct_hmp (Tensor): Center heatmap for current level with
                shape (N, num_classes, H, W).
            ct_off (Tensor): Center offset for current level with
                shape (N, corner_offset_channels, H, W).
            tracking_off: shape (N, tracking_offset_channels, H, W).
            ct_ltrb: shape (N, ct_ltrb_channels, H, W).
            targets: dict.

        Returns:
            pass
        """
        gt_ct_hmp = targets['center_heatmap']
        gt_ct_off = targets['center_offset']
        gt_tracking_off = targets['center_tracking_offset']
        gt_ct_ltrb = targets['center_ltrb']

        # Detection loss
        det_loss = self.loss_heatmap(
            ct_hmp.sigmoid(),
            gt_ct_hmp,
            avg_factor=max(1,
                           gt_ct_hmp.eq(1).sum()))

        # Offset loss
        # We only compute the offset loss at the real corner position.
        # The value of real corner would be 1 in heatmap ground truth.
        # The mask is computed in class agnostic mode and its shape is
        # batch * 1 * width * height.
        # !!!here need to make the mask correct, loss can be adjust by config.
        ct_pos_mask = gt_ct_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_ct_hmp)
        off_loss = self.loss_offset(
            ct_off,
            gt_ct_off,
            ct_pos_mask,
            avg_factor=max(1, ct_pos_mask.sum()))

        # Tracking Loss
        tracking_loss = self.loss_offset(
            tracking_off,
            gt_tracking_off,
            ct_pos_mask,
            avg_factor=max(1, ct_pos_mask.sum()))

        # LTRB Amodel Loss
        ltrb_loss = self.loss_offset(
            ct_ltrb,
            gt_ct_ltrb,
            ct_pos_mask,
            avg_factor=max(1, ct_pos_mask.sum()))


        return det_loss, off_loss, tracking_loss, ltrb_loss

    def process(self, x, img_metas, pre_inds, rescale=False):
        """Inference process function in CenterTrack

        Args:
            x (list[dict or list]): Backbone output.
            pre_inds (Tensor): Previous frame tracks center index with shape (1, K).
            // pre_inds is not used in current version.
        """
        # x = x[-1]   # last stage output
        if len(x) > 1:
            warnings.warn('Current will only use last level output!')
        for i in range(len(x)):
            x[i]['hm'] = x[i]['hm'].sigmoid_()

        new_x = {k: [x[i][k] for i in range(len(x))] for k in x[0].keys()}
        # Decode output, for different image
        result_list = self.get_bboxes(
            new_x['hm'], new_x['reg'], new_x['ltrb_amodal'],
            new_x['tracking'], img_metas, rescale=rescale)

        return result_list


    def get_bboxes(self,
                   ct_heats,
                   ct_offs,
                   ltrb_amodals,
                   trackings,
                   img_metas,
                   rescale=False,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.
        The outlier list represent different levels.

        Args:
            ct_heats (list[Tensor]): Center heatmaps for each level
                with shape (N, num_classes, H, W).
            ct_offs (list[Tensor]): Center offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert ct_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    ct_heats[-1][img_id:img_id + 1, :],
                    ct_offs[-1][img_id:img_id + 1, :],
                    ltrb_amodals[-1][img_id:img_id + 1, :],
                    trackings[-1][img_id:img_id + 1, :],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list

    def _get_bboxes_single(self,
                           ct_heat,
                           ct_off,
                           ltrb_amodal,
                           tracking,
                           img_meta,
                           rescale=False,
                           with_nms=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            ct_heat (Tensor): Center heatmap for current level with
                shape (N, num_classes, H, W).
            ct_off (Tensor): Center offset for current level with
                shape (N, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]

        ret = self.decode_heatmap(
            ct_heat=ct_heat,
            ct_off=ct_off,
            ltrb_amodal=ltrb_amodal,
            tracking=tracking,
            img_meta=img_meta,
            k=self.test_cfg.center_topk,
            kernel=self.test_cfg.local_maximum_kernel)

        # scale_factor is rate between input shape and ori_img shape, so div it can get original img coord.
        # CenterNet has no NMS.
        if rescale:
            scale_key = 'first_scale_factor' if 'first_scale_factor' in img_meta else 'scale_factor'
            for k in ['bboxes_amodal', 'bboxes', 'cts', 'tracking']:
                ret[k] = ret[k]/ret[k].new_tensor(img_meta[scale_key][..., :ret[k].shape[-1]])
        # print(ret['tracking'][:, :10], "Fin")

        scores = ret['scores'].clone().view([-1, 1])    # clone to prevent multiple operation.

        idx = scores.argsort(dim=0, descending=True)
        scores = scores[idx].view(-1)   # length = len(idx)
        keepinds = (scores > self.test_cfg.out_thresh)
        for k in ret:
            ret[k] = ret[k].view(-1, ret[k].shape[-1])[idx]
            ret[k] = ret[k].view(-1, ret[k].shape[-1]).squeeze(-1)
            ret[k] = ret[k][keepinds]
        # ret['clses'] = ret['clses'] + 1

        detections = torch.cat([ret['bboxes'], ret['scores'].unsqueeze(-1)], -1)      # cat use clone()
        ret['detections'] = detections

        return ret

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels
        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.
            mask (Tensor | None): Mask of featuremap. Default: None.

        Returns:
            feat (Tensor): Gathered feature.
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).repeat(1, 1, dim)
        feat = feat.gather(1, ind)      # N,K,C
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _transpose_and_gather_feat(self, feat, ind):
        """Transpose and gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.

        Returns:
            feat (Tensor): Transposed and gathered feature.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, k=20):
        """Get top k positions from heatmap.
        In CenterTrack, it first get topk scores and inds for each category,
        then get topk for these CK scores and inds. This implement difference
        not take effect for MOT dataset.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        # TODO: try different implement.
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def decode_heatmap(self,
                       ct_heat,
                       ct_off,
                       ltrb_amodal,
                       tracking,
                       img_meta=None,
                       k=100,
                       kernel=3,
                       num_dets=1000):
        """Transform outputs for a single batch item into raw bbox predictions.

        Args:
            ct_heat (Tensor): Center heatmap for current level with
                shape (N, num_classes, H, W).
            ct_off (Tensor): Center offset for current level with
                shape (N, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            k (int): Get top k corner keypoints from heatmap.
            kernel (int): Max pooling kernel for extract local maximum pixels.
            num_dets (int): Num of raw boxes before doing nms.

        Returns:
            tuple[torch.Tensor]: Decoded output of CornerHead, containing the
            following Tensors:

            - bboxes (Tensor): Coords of each box.
            - scores (Tensor): Scores of each box.
            - clses (Tensor): Categories of each box.
        """
        batch, _, height, width = ct_heat.size()
        inp_h, inp_w, _ = img_meta['pad_shape'] # assume use pad as last size transform

        # perform nms on heatmaps
        # print("In head ", tracking.shape)
        # for hi in range(60, 70):
        #     for wi in range(180,190):
        #         print(hi, wi, tracking[..., hi, wi], ct_heat[..., hi, wi]) 
        # _ = input()

        ct_heat = self._local_maximum(ct_heat, kernel=kernel)

        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(ct_heat, k=k)

        # In Centertrack, xs/ys/cts is in feature coord space, not add offset.
        cts = torch.cat([ct_xs.unsqueeze(2), ct_ys.unsqueeze(2)], dim=2)
        ret = {'scores': ct_scores.unsqueeze(-1), 'clses': ct_clses.float().unsqueeze(-1)}

        # We use repeat instead of expand here because expand is a
        # shallow-copy function. Thus it could cause unexpected testing result
        # sometimes. Using expand will decrease about 10% mAP during testing
        # compared to repeat.
        # For CenterNet, it is not necessary to construct coord pair.
        # ct_xs, ct_ys -> batch x k
        ct_off = self._transpose_and_gather_feat(ct_off, ct_inds)   # batch,c,h,w -> batch,k,c | c=2

        ct_xs = ct_xs + ct_off[..., 0]
        ct_ys = ct_ys + ct_off[..., 1]  # 0-x, 1-y as get_target()

        ltrb_amodal = self._transpose_and_gather_feat(ltrb_amodal, ct_inds) # B x K x 4
        ltrb_amodal = ltrb_amodal.view(batch, k, 4)
        tl_xs = ct_xs + ltrb_amodal[..., 0]
        tl_ys = ct_ys + ltrb_amodal[..., 1]
        br_xs = ct_xs + ltrb_amodal[..., 2]
        br_ys = ct_ys + ltrb_amodal[..., 3]

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=2)

        tracking = self._transpose_and_gather_feat(tracking, ct_inds)

        # all possible boxes based on top k corners (ignoring class)
        # To input coord space
        rate_w = (inp_w / width)
        rate_h = (inp_h / height)

        # To original image coord space(w/o rescale)
        # assert 'border' not in img_meta, "Not support test with crop now!"
        # x_off, y_off = 0, 0
        if 'first_scale_factor' in img_meta:
            last_scale = bboxes.new_tensor(img_meta['scale_factor'])
        else:
            last_scale = None
        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]
        
        bboxes = self._to_img_space(bboxes, rate_w, rate_h, x_off, y_off, last_scale)
        ret['bboxes_amodal'] = bboxes.clone()
        ret['bboxes'] = bboxes.clone()
        ret['cts'] = self._to_img_space(cts, rate_w, rate_h, x_off, y_off, last_scale)

        # The target is given on feature shape, for tracking, border not influence
        # so x_off=y_off=0.
        # print(tracking[:, :10], "Before")
        ret['tracking'] = self._to_img_space(
            tracking, rate_w, rate_h, last_scale=last_scale, mask_neg=False)
        # print(ret['tracking'][:, :10], "Mid")
        return ret

    def _to_img_space(
        self, x, rate_w, rate_h, x_off=0, y_off=0, last_scale=None, mask_neg=True):
        if x.shape[-1] == 4:        # boxes
            x[..., [0,2]] *= rate_w
            x[..., [1,3]] *= rate_h
            if last_scale is not None:
                x = x/last_scale
            x[..., [0,2]] -= x_off
            x[..., [1,3]] -= y_off
            if mask_neg:
                x *= x.gt(0.0).type_as(x)
        elif x.shape[-1] == 2:      # points offset
            x[..., [0]] *= rate_w
            x[..., [1]] *= rate_h
            if last_scale is not None:
                x = x/last_scale[..., :2]
            x[..., [0]] -= x_off
            x[..., [1]] -= y_off
            if mask_neg:
                x *= x.gt(0.0).type_as(x)
        else:
            raise ValueError("Get unexpected input tensor with shape {}".format(x.shape))
        
        return x