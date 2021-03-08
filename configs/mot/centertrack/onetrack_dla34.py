_base_ = [
    './mot_challenge.py',
]
model = dict(
    type='OneTrack',
    backbone=dict(
        type='DLASeg',
        num_layers=34,
        heads=dict(hm=1, tracking=2, ltrb_amodal=4),
        head_convs=dict(hm=[256], tracking=[256], ltrb_amodal=[256]),
        opt_base=dict(pre_img=True, pre_hm=True,),
        opt=dict(
            dla_node='dcn',
            # Prior probability 0.01, for stabilize training in imbalanced classes.
            prior_bias=-4.6,
            head_kernel=3,
            model_output_list=False,
            )),
    head=dict(
        type='OneTrackHead',
        num_classes=1,
        num_feat_levels=1,
        loss_heatmap=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     alpha=0.25,
                     gamma=2.0,
                     loss_weight=1),
        loss_giou=dict(type='GIoULoss', loss_weight=1),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1),
        matcher=dict(
            alpha=0.25, gamma=2.0,
            cost_class=1, cost_bbox=1, cost_giou=1,
        )
    ),
    # 1 is not retain at all, for CenterTrack, 32, 10, 1 can be test...
    tracker=dict(type='CenterTrackTracker', hungarian=False, momentums=None, num_frames_retain=10),
    pretrains=dict(
        backbone= ('./experiments/CenterTrack/checkpoints/crowdhuman.pth', 'default'),
        # backbone= ('./experiments/CenterTrack/checkpoints/mot17_fulltrain.pth', 'mmcv'),
    ),
    train_cfg=dict(
        dense_head=dict()
    ),
    test_cfg=dict(
        tracker=dict(pre_thresh=0.5, new_thresh=0.4),
        dense_head=dict(center_topk=100, local_maximum_kernel=3, out_thresh=0.4)
    ))
# FP16
fp16 = dict(loss_scale=512.)
# optimizer
optimizer = dict(type='Adam', lr=0.000125)    # 0.000125-bs32
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[60])      # 3 in original mot==
# runtime settings
total_epochs = 70
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
