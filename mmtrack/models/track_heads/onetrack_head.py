from math import ceil, log

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, bbox_overlaps
from mmdet.models.builder import HEADS, build_loss
from ..mot.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

class MinCostMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, alpha, gamma, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss_alpha = alpha
        self.focal_loss_gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, hm, pred_boxes, gt_labels, gt_bboxes, img_shape):
        """ Performs the matching

        Params:
            hm: Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            pred_boxes: Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            gt_labels: Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                        objects in the target) containing the class labels
            gt_boxes: Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            img_shape: Tensor of dim[2] represent the shape of input images

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
        bs, k, h, w = hm.shape
        img_h, img_w = img_shape[:2]
        img_shape_xyxy = hm.new_tensor([img_w, img_h, img_w, img_h])[None]

        # We flatten to compute the cost matrices in a batch
  
        batch_out_prob = hm.permute(0, 2, 3, 1).reshape(bs, h*w, k).sigmoid() # [batch_size, num_queries, num_classes]
        batch_out_bbox = pred_boxes.permute(0, 2, 3, 1).reshape(bs, h*w, 4) # [batch_size, num_queries, 4]
        
        indices = []
        
        
        for i in range(bs):
            tgt_ids = gt_labels[i]
            
            if tgt_ids.shape[0] == 0:
                indices.append((torch.as_tensor([]).to(batch_out_prob), torch.as_tensor([]).to(batch_out_prob)))
                continue
                
            tgt_bbox = gt_bboxes[i]
            out_prob = batch_out_prob[i] 
            out_bbox = batch_out_bbox[i] 
            
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            # Here assume same image size of all samples in one batch
            # This is make sure by dataset pipeline
            out_bbox_ = out_bbox / img_shape_xyxy
            tgt_bbox_ = tgt_bbox / img_shape_xyxy
            cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

            # Compute the giou cost betwen boxes
            # bbox_overlaps support batch (..., Bn), n can be 0.
            cost_giou = -bbox_overlaps(out_bbox, tgt_bbox, mode='giou')
            # cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            
            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(tgt_ids)).to(src_ind)
            indices.append((src_ind, tgt_ind))
        
        # bs - (index_i, index_j)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

@HEADS.register_module()
class OneTrackHead(BaseDenseHead):
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
                     type='FocalLoss',
                     use_sigmoid=True,
                     alpha=0.25,
                     gamma=2.0,
                     loss_weight=1),
                 loss_giou=dict(
                     type='GIoULoss', loss_weight=1),
                 loss_offset=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1),
                 matcher=dict(
                     alpha=0.25, gamma=2.0,
                     cost_class=1, cost_bbox=1, cost_giou=1,
                 )):
        super(OneTrackHead, self).__init__()
        self.num_classes = num_classes
        self.corner_offset_channels = 2
        self.num_feat_levels = num_feat_levels
        self.loss_heatmap = build_loss(
            loss_heatmap) if loss_heatmap is not None else None
        self.loss_giou = build_loss(
            loss_giou) if loss_giou is not None else None
        self.loss_offset = build_loss(
            loss_offset) if loss_offset is not None else None
        self.matcher = MinCostMatcher(**matcher)
        self.fp16_enabled = False
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

    def get_indices_and_bboxes(self,
                    gt_bboxes_amodal,
                    gt_labels,
                    ct_heats,
                    pred_ltrb,
                    img_shape,
                    pre_cts,):
        """Generate label assignment indices.
        In dla34, it return list[dict](default), which contain all
        desired output(e.g. reg/tracking/ltrb_amodel...).
        So, this head only generate target for this output.

        TODO: Now, here generate shared target for all levels, 
        whicht is undesired when num_levels > 1, due to 
        the label assignment is not same.
        """
        loc_ft_coord = self._locations(ct_heats)[None]
        pred_boxes = self._apply_ltrb(loc_ft_coord, pred_ltrb)
        indices = self.matcher(
            ct_heats, pred_boxes, gt_labels, gt_bboxes_amodal, img_shape[:2])

        return indices, pred_boxes
    
    @force_fp32(apply_to=('ct_heats', 'tracking_offs', 'ct_ltrbs'))
    def loss(self,
             ct_heats,
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
        indices, pred_boxes = self.get_indices_and_bboxes(
            gt_bboxes_amodal,
            gt_labels,
            ct_heats[-1],
            ct_ltrbs[-1],       # Now only generate shared indices from last level.
            img_metas[0]['pad_shape'],
            pre_cts)
        mlvl_indices = [indices for _ in range(self.num_feat_levels)]
        mlvl_pred_boxes = [pred_boxes for _ in range(self.num_feat_levels)]
        gt_dict = dict(
            gt_labels=gt_labels, 
            gt_bboxes_amodal=gt_bboxes_amodal,
            gt_match_indices=gt_match_indices,
            pre_cts=pre_cts,
            img_shape=img_metas[0]['pad_shape'])
        mlvl_targets = [gt_dict for _ in range(self.num_feat_levels)]
        det_losses, giou_losses, l1_losses, tracking_losses = multi_apply(
            self.loss_single, ct_heats, tracking_offs, mlvl_pred_boxes, 
            mlvl_targets, mlvl_indices)
        loss_dict = dict(
            det_loss=det_losses, giou_loss=giou_losses, tracking_loss=tracking_losses, l1_loss=l1_losses)
        return loss_dict

    def loss_single(
        self, ct_hmp, tracking_off, pred_boxes, 
        targets, indices):
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
            tracking_off: shape (N, tracking_offset_channels, H, W).
            ct_ltrb: shape (N, ct_ltrb_channels, H, W).
            targets: dict.

        Returns:
            pass
        """
        # Focal Loss here, so not need to add .sigmoid()
        det_loss = self.loss_labels(
            ct_hmp, targets['gt_labels'], indices)
        giou_loss, l1_loss = self.loss_boxes(
            pred_boxes, targets['gt_bboxes_amodal'], 
            indices, targets['img_shape'])
        tracking_loss = self.loss_tracking(
            tracking_off, targets['gt_bboxes_amodal'], 
            targets['gt_match_indices'],
            targets['pre_cts'], targets['img_shape'], indices)

        return det_loss, giou_loss, l1_loss, tracking_loss

    def loss_labels(self, src_logits, labels, indices):
        """
        labels: list[Tensor], corr to bs, bbox_labels
        indices: list[tuple(Tensor)], corr to bs, (pred_inds, gt_inds)
        """
        bs, k, h, w = src_logits.shape
        src_logits = src_logits.permute(0, 2, 3, 1).reshape(bs, h*w, k)
        
        # idx is tuple consist of (batch_idx, src_idx)
        # the size corr to the sum of all idxs in different size
        # bs + h*w -> one dim
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(labels, indices)])
        # val = num_classes will fill zero in flollow
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)   # bs*h*w, k
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)   # bs*h*w
        # !! This is not need for focal loss not support one-hot label
        # pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        # labels = torch.zeros_like(src_logits).long()
        # labels[pos_inds, target_classes[pos_inds]] = 1

        # comp focal loss.
        # Positive sample nums = pos_inds.sum() ~ target_classes_o.shape[0]
        # if max_cost_assign can be one-to-one
        # negative sample nums = others of b*h*w map
        num_pos_samples = (target_classes != self.num_classes).sum()
        det_loss = self.loss_heatmap(
            src_logits,
            target_classes,
            avg_factor=max(1, num_pos_samples))
        return det_loss

    def loss_boxes(self, src_boxes, gt_bboxes_amodal, indices, img_shape):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        
        bs, k, h, w = src_boxes.shape
        src_boxes = src_boxes.permute(0, 2, 3, 1).reshape(bs, h*w, k)
        
        src_boxes = src_boxes[idx]
        target_boxes = torch.cat(
            [t[i] for t, (_, i) in zip(gt_bboxes_amodal, indices)], dim=0)

        # is_aligned = True and all positive sample here
        loss_giou = self.loss_giou(
            src_boxes, target_boxes,
            avg_factor=max(1, target_boxes.shape[0]))

        img_h, img_w = img_shape[:2]
        img_shape_xyxy = src_boxes.new_tensor([img_w, img_h, img_w, img_h])[None]
        src_boxes_ = src_boxes / img_shape_xyxy
        target_boxes_ = target_boxes / img_shape_xyxy

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses_l1 = loss_bbox.sum() / target_boxes.shape[0]

        return loss_giou, losses_l1

    def loss_tracking(
        self, src_tk_offs, gt_bboxes, gt_match_indices, 
        pre_cts, img_shape, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # Generate Tracking Offset
        batch_size, _, height, width = src_tk_offs.shape
        img_h, img_w = img_shape[:2]

        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)
        assert int(img_h/height) == int(img_w/width) == 4, "Not as desired"

        gt_ct_tracking = []
        for batch_id in range(batch_size):
            # Ground truth of corner embedding per image is a list of coord set
            cur_ct_tracking = []
            for box_id in range(len(gt_bboxes[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0

                # Use coords in the feature level to generate ground truth
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio

                # Int coords on feature map/ground truth tensor
                center_x_idx = int(min(scale_center_x, width - 1))
                center_y_idx = int(min(scale_center_y, height - 1))

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
                    cur_ct_tracking.append(center_x.new_tensor(
                        [center_x_pre_offset, center_y_pre_offset]))
                else:
                    cur_ct_tracking.append(center_x.new_zeros(2))
            if len(cur_ct_tracking):
                cur_ct_tracking = torch.stack(cur_ct_tracking, dim=0)
            else:
                cur_ct_tracking = center_x.new_zeros((0, 2))
            gt_ct_tracking.append(cur_ct_tracking)

        idx = self._get_src_permutation_idx(indices)
        
        bs, k, h, w = src_tk_offs.shape
        src_tk_offs = src_tk_offs.permute(0, 2, 3, 1).reshape(bs, h*w, k)
        
        src_tk_offs = src_tk_offs[idx]
        target_tk_offs = torch.cat(
            [t[i] for t, (_, i) in zip(gt_ct_tracking, indices)], dim=0)

        # is_aligned = True here
        # All positive, so no weight here!
        loss_tracking = self.loss_offset(
            src_tk_offs, target_tk_offs, 
            avg_factor=max(1, src_tk_offs.shape[0])
        )

        return loss_tracking

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
            new_x['hm'], new_x['ltrb_amodal'],
            new_x['tracking'], img_metas, rescale=rescale)

        return result_list

    @force_fp32(apply_to=('ct_heats', 'ltrb_amodals', 'trackings'))
    def get_bboxes(self,
                   ct_heats,
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
                    ltrb_amodals[-1][img_id:img_id + 1, :],
                    trackings[-1][img_id:img_id + 1, :],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list

    def _get_bboxes_single(self,
                           ct_heat,
                           ltrb_amodal,
                           tracking,
                           img_meta,
                           rescale=False,
                           with_nms=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            ct_heat (Tensor): Center heatmap for current level with
                shape (N, num_classes, H, W).
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

    def _apply_ltrb(self, locations, pred_ltrb): 
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W) 
        """

        pred_boxes = torch.zeros_like(pred_ltrb)
        # !! ltrb is l/t/r/b - center, which is different from OneNet
        # here are all add operations.
        pred_boxes[:,0,:,:] = locations[:,0,:,:] + pred_ltrb[:,0,:,:]  # x1
        pred_boxes[:,1,:,:] = locations[:,1,:,:] + pred_ltrb[:,1,:,:]  # y1
        pred_boxes[:,2,:,:] = locations[:,0,:,:] + pred_ltrb[:,2,:,:]  # x2
        pred_boxes[:,3,:,:] = locations[:,1,:,:] + pred_ltrb[:,3,:,:]  # y2

        return pred_boxes    
    
    @torch.no_grad()
    def _locations(self, features, stride=4):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2            
        
        locations = locations.reshape(h, w, 2).permute(2, 0, 1)
        
        return locations

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

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

        # Not support NMS now
        # ct_heat = self._local_maximum(ct_heat, kernel=kernel)

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
        rate_w = (inp_w / width)
        rate_h = (inp_h / height)

        ltrb_amodal = self._transpose_and_gather_feat(ltrb_amodal, ct_inds) # B x K x 4
        ltrb_amodal = ltrb_amodal.view(batch, k, 4)
        tl_xs = ct_xs*rate_w + ltrb_amodal[..., 0]
        tl_ys = ct_ys*rate_h + ltrb_amodal[..., 1]
        br_xs = ct_xs*rate_w + ltrb_amodal[..., 2]
        br_ys = ct_ys*rate_h + ltrb_amodal[..., 3]

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=2)

        tracking = self._transpose_and_gather_feat(tracking, ct_inds)

        # To original image coord space(w/o rescale)
        # assert 'border' not in img_meta, "Not support test with crop now!"
        # x_off, y_off = 0, 0
        if 'first_scale_factor' in img_meta:
            last_scale = bboxes.new_tensor(img_meta['scale_factor'])
        else:
            last_scale = None
        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]
        
        bboxes = self._to_img_space(
            bboxes, x_off=x_off, y_off=y_off, last_scale=last_scale)
        ret['bboxes_amodal'] = bboxes.clone()
        ret['bboxes'] = bboxes.clone()
        center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2.0
        center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2.0
        ret['cts'] = torch.stack([center_x, center_y], dim=-1)

        # The target is given on feature shape, for tracking, border not influence
        # so x_off=y_off=0.
        # print(tracking[:, :10], "Before")
        ret['tracking'] = self._to_img_space(
            tracking, rate_w, rate_h, last_scale=last_scale, mask_neg=False)
        # print(ret['tracking'][:, :10], "Mid")
        return ret

    def _to_img_space(
        self, x, rate_w=1, rate_h=1, x_off=0, y_off=0, last_scale=None, mask_neg=True):
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