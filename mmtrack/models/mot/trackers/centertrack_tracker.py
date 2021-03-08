import torch
import copy
import numpy as np
import mmcv
import cv2
from mmdet.core import bbox_overlaps, multiclass_nms
from scipy.optimize import linear_sum_assignment
from math import ceil, log

from mmtrack.core import imrenormalize
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker
from ..utils import gaussian_radius, gen_gaussian_target


@TRACKERS.register_module()
class CenterTrackTracker(BaseTracker):
    """Tracker for CenterTrack.

    Args:
        hungarian (bool, optional): Whether to use hungarian algorithm to
            match detections and tracks.
    """

    def __init__(self,
                 test_cfg,
                 hungarian=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.test_cfg = test_cfg
        self.hungarian = hungarian

    def prepare_heatmap(self, frame_id, img_metas, device, with_hm=True, rescale=False):
        if not self.empty:
            memo = self.memo      
            # bboxes corr to original image space in test with rescale == True
            # To project to input space, here need to add transform
            # first del border, then rescale
            # Due to the usage of cat function, the tensor is clone from buffer                  
            bboxes = memo.bboxes[memo.frame_ids == frame_id - 1]
            # TODO: support different transform for bboxes in different frames
            # TODO: support crop transform
            if rescale:
                img_meta = img_metas[0]
                x_off = img_meta['border'][2]
                y_off = img_meta['border'][0]
                if 'first_scale_factor' in img_metas[0]:
                    bboxes *= torch.tensor(img_meta['first_scale_factor']).to(
                        bboxes.device)
                    bboxes[..., [0,2]] += x_off
                    bboxes[..., [1,3]] += y_off
                    bboxes *= torch.tensor(img_meta['scale_factor']).to(
                        bboxes.device)
                else:

                    bboxes *= torch.tensor(img_metas[0]['scale_factor']).to(
                        bboxes.device)
                    bboxes[..., [0,2]] += x_off
                    bboxes[..., [1,3]] += y_off
        else:
            bboxes = []
        # print(bboxes, "pre_hm_inp")
        height, width = img_metas[0]['pad_shape'][0], img_metas[0]['pad_shape'][1]    # input shape
        input_hm = torch.zeros((1, height, width), device=device)
        output_inds = []

        
        for bbox in bboxes:
            if bbox[-1] < self.test_cfg.pre_thresh:
                continue
            box_width, box_height = ceil(bbox[3] - bbox[1]), ceil(bbox[2] - bbox[0])
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            center_x_idx = int(min(center_x, width - 1))
            center_y_idx = int(min(center_y, height - 1))
            
            radius = gaussian_radius((box_height, box_width))
            radius = max(0, int(radius))
            if with_hm:
                input_hm[0] = gen_gaussian_target(
                        input_hm[0], [center_x_idx, center_y_idx],
                        radius)
            # use same int center as input_hm
            output_inds.append(center_y_idx * width + center_x_idx)
        
        # self._show_heatmap(img_metas[0]['ori_filename'], input_hm)
        input_hm = input_hm[None]
        output_inds = torch.tensor(output_inds, dtype=torch.long, device=device) # deepcopy, diff from from_numpy
        # TODO: support aug_test
        return input_hm, output_inds

    def track(self,
              dets,
              det_cts,
              det_cats,
              tracking_offs,
              frame_id,
              img_metas=None,
              public_det=None,
              **kwargs):
        """Tracking forward function.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            dets (Tensor): of shape (N, 5).
            det_cats (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.

        Returns:
            tuple: Tracking results.
        """

        if self.empty:
            num_new_tracks = dets.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
        else:
            # Prepare detections
            # print(tracking_offs.shape, tracking_offs)
            # print(det_cts.shape, det_cts)
            # print(img_metas[0]['filename'])
            # _ = input()

            det_pre_cts = det_cts + tracking_offs
            memo = self.memo                                            # memo(@property) return buffers for keys in memo_items, fetch from tracks. 
            tracks = memo.bboxes        # contain inactive but not popped tracks, (m, 5), m=num_tracks
            track_ids = memo.ids
            track_sizes = (tracks[:, 2]-tracks[:, 0])*(tracks[:, 3]-tracks[:, 1])    # m
            track_cats = memo.labels
            track_cts = memo.cts    # (m, 2)

            det_sizes = (dets[:, 2]-dets[:, 0])*(dets[:, 3]-dets[:, 1])  # n
            # det_cats
            n, m = det_pre_cts.size(0), track_cts.size(0)
            dist = ((track_cts.view(1, -1, 2)-det_pre_cts.view(-1, 1, 2))**2).sum(dim=2)    # (n, m)

            # print("tracks ", m)
            # track_fids = memo.frame_ids
            # for tid, tct, tbbox, tfid in zip(track_ids, track_cts, tracks, track_fids):
            #     print(tid, tct, tbbox, tfid, sep='-')
            # print("dets", n)
            # for dct, doff, dbbox in zip(det_cts, tracking_offs, dets):
            #     print(dct, doff, dbbox[2]-dbbox[0], dbbox[3]-dbbox[1])
            invalid = (
                (dist > track_sizes.view(1, m)) |
                (dist > det_sizes.view(n, 1)) |
                (det_cats.view(n, 1) != track_cats.view(1, m))
            )
            dist = dist + invalid.float()*1e18

            if self.hungarian:
                dist[dist > 1e18] = 1e18
                row, col = linear_sum_assignment(dist.cpu().numpy())
            else:
                # greedy assignment
                matched_indices = self._greedy_assignment(copy.deepcopy(dist))
                row, col = matched_indices[:, 0], matched_indices[:, 1]
            unmatched_det_ids = [d for d in range(n) if not (d in row)]
            unmatched_track_ids = [d for d in range(m) if not (d in col)]
                
            if self.hungarian:
                matched_ids = []
                for m in zip(row, col):
                    if dist[m[0], m[1]] > 1e16:
                        unmatched_det_ids.append(m[0])
                        unmatched_track_ids.append(m[1])
                    else:
                        matched_ids.append(m)
                matched_ids = np.array(matched_ids).reshape(-1, 2)  # 2->n|m
            else:
                matched_ids = matched_indices

            # print('matched_ids -> ')
            # for r, c in matched_ids:
            #     print(r, c, det_cts[r], det_pre_cts[r], track_cts[c], track_ids[c])

            # matched dets

            # unmatched dets as new tracks
            if public_det is not None and len(unmatched_det_ids) > 0:
                raise NotImplementedError("Not implement for public detection")
            else:
                # Private detection: create tracks for all un-matched detections
                unmatched_dets = dets[unmatched_det_ids]    # (n', 5)
                valid_new_mask = unmatched_dets[:, 4] > self.test_cfg.new_thresh
                unmatched_det_ids = torch.tensor(
                    unmatched_det_ids, dtype=torch.int64)[valid_new_mask]

            ids = torch.arange(
                    self.num_tracks,
                    self.num_tracks + unmatched_det_ids.shape[0],
                    dtype=torch.long)
            self.num_tracks += unmatched_det_ids.shape[0]

            dets = torch.cat((dets[matched_ids[:, 0]], dets[unmatched_det_ids]), dim=0)
            det_cats = torch.cat((
                det_cats[matched_ids[:, 0]], det_cats[unmatched_det_ids]), dim=0)
            det_cts = torch.cat((det_cts[matched_ids[:, 0]], det_cts[unmatched_det_ids]), dim=0)
            ids = torch.cat((track_ids[matched_ids[:, 1]], ids), dim=0)
            # print(dets, det_cts, "input_dets")

        self.update(
            ids=ids,
            bboxes=dets[:, :4],
            scores=dets[:, -1],
            labels=det_cats,
            cts=det_cts,
            frame_ids=frame_id)

        return dets, det_cats, ids

    def _greedy_assignment(self, dist):
        matched_indices = []
        if dist.shape[1] == 0:
            return np.array(matched_indices, np.int32).reshape(-1, 2)
        for i in range(dist.shape[0]):
            j = dist[i].argmin()
            if dist[i][j] < 1e16:
                dist[:, j] = 1e18
                matched_indices.append([i, j])
        return np.array(matched_indices, np.int32).reshape(-1, 2)

    def _show_heatmap(self, pre_img, pre_hm, to_pre=True):
        """
        pre_hm: (1, h, w)
        pre_img: str
        """
        if to_pre:
            img_id = int(pre_img.split('/')[-1].split('.')[0])
            if img_id <= 1:
                print("First frame do not has pre_frame")
                return
            pre_img = pre_img.replace(
                "{:06d}".format(img_id), "{:06d}".format(img_id-1))

        pre_img_name = pre_img
        pre_img = mmcv.imread(pre_img)
        pre_hm = (pre_hm.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
        pre_hm = cv2.applyColorMap(pre_hm, cv2.COLORMAP_JET)
        pre_img = mmcv.imresize_like(pre_img, pre_hm)
        out_img = (pre_img + pre_hm)/255
        out_img = out_img/np.max(out_img)
        out_img = (out_img*255).astype(np.uint8)

        img_name = pre_img_name.split('/')[-1]
        out_file = 'data/experiments/CenterTrack/debug/hm_{}'.format(img_name)
        mmcv.imwrite(out_img, out_file)
        print('Show heatmap', out_file)
        # _ = input()
