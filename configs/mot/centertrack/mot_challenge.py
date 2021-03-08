# dataset settings
dataset_type = 'MOTChallengeDataset'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# to rgb = False will not change BGR to RGB
img_norm_cfg = dict(
    mean=[104.0136177, 114.0342201, 119.91659325], 
    std=[73.6027614, 69.8908182, 70.91507925], to_rgb=False)
# auto_crop_mode='max', which means max(h, w) will as the width of cropped image.
# if the as_ratio is w<h, these may lead to small cropped region.
# These can be solved by change max(h, w) to w.
# if auto_crop_mode='max', then crop_size will be modified by the input image size.
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCenterCropPad',
        crop_size=(544, 960),       # means h, w
        auto_crop_mode='max',       # For max crop in CenterTrack
        share_params=True,
        filter_coi_boxes=False,     # For amodel bbox in CenterTrack
        bbox_clip_border=True,     # For amodel bbox in CenterTrack
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        test_mode=False,
        test_pad_mode=None,
        **img_norm_cfg),
    dict(
        type='SeqResize', 
        img_scale=(960, 544),       # means w, h
        share_params=True, 
        keep_ratio=False, 
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(
        type='CTrackTarget',
        opt=dict(
            reutrn_hm=True,
            down_ratio=4,
            hm_disturb=0.05,
            lost_disturb=0.4,
            fp_disturb=0.1)),
    dict(
        type='MatchInstances', 
        skip_nomatch=True,
        match_keys=('gt_instance_ids', 'disturb_ref_ids')),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids', 'pre_hm', 'pre_cts', 'gt_bboxes_amodal'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize'),
            dict(
                type='SeqRandomCenterCropPad',
                crop_size=(544, 960),
                auto_crop_mode='fix_res_max',
                ratios=None,
                border=None,
                share_params=False,
                test_mode=True,
                # test_pad_mode=['logical_or', 127],
                **img_norm_cfg),    # img_norm_cfg is used for pad the cropped_img
            dict(type='Resize', img_scale=(960, 544), keep_ratio=False, override=True),
            # if double resize, the scale factor is related to new_size and img_size, where the latter is not the
            # original image size.
            # keep_ratio , here make no difference with True/False because resize and rescale have same scale_factor
            # scale_factor is used to recover scale, but scale not, for it may corr to rescale, which means
            # max long edge and max short edge.
            # when call resize, scale and scale_factor will both be updated, if it both exist, 
            # means resize func has been called before, if override=True, it will be deleted, otherwise raise Error.
            # Here will get part image when as-rate<default, e.g. 640x480 in MOT17
            # Another choice like follow, only one Resize:
            # dict(type='Resize'),
            # dict(
            #     type='RandomCenterCropPad',
            #     crop_size=None,
            #     ratios=None,
            #     border=None,
            #     test_mode=True,
            #     test_pad_mode=['logical_or', 127],
            #     **img_norm_cfg),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'], meta_keys=('border', 'first_scale_factor'))
        ])
]
data_root = 'data/MOT17/'
data = dict(
    samples_per_gpu=8,     # 2 gpus, each gpu with 16 bs, 2 for debug
    workers_per_gpu=2,     # default to 2
    train=dict(
        type=dataset_type,
        visibility_thr=-1,
        ann_file=data_root + 'annotations/half-train_cocoformat_FRCNN.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=3,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat_FRCNN.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat_FRCNN.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline))
