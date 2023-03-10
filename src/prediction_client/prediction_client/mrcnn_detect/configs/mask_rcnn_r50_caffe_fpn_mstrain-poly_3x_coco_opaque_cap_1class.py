_base_ = '/home/xjgao/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1),
        mask_head=dict( 
            num_classes=1,
            )),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            )))
optimizer = dict(lr=5e-4)
lr_config = dict(warmup=None)

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='MinIoURandomCrop',min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3,  bbox_clip_border=True),
    dict(type='Rotate', level=1, scale=1, center=None, img_fill_val=128, seg_ignore_label=255, prob=0.5, max_rotate_angle=30, random_negative_prob=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoDataset'
dataset_train_A = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/real_data_1000_grey_box/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/real_data_1000_grey_box',
        pipeline=train_pipeline,
        classes = ('object',))
dataset_train_B = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/realdata/Desktop/mix/0/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/realdata/Desktop/mix/0',
        pipeline=train_pipeline,
        classes = ('object',))
dataset_train_C = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/Labeled_Autostore_8.2/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/Labeled_Autostore_8.2',
        pipeline=train_pipeline,
        classes = ('object',))
dataset_train_D = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/virtual_data_9.1/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/virtual_data_9.1',
        pipeline=train_pipeline,
        classes = ('object',))
dataset_train_E = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/Labeled_Autostore_9.5/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/Labeled_Autostore_9.5',
        pipeline=train_pipeline,
        classes = ('object',))

dataset_train_A1 = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/45/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/45',
        pipeline=train_pipeline,
        classes = ('object',))
dataset_train_A1_rp = dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_train_A1)        
dataset_train_B1 = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/0420/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/0420',
        pipeline=train_pipeline,
        classes = ('object',))
dataset_train_B1_rp = dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_train_B1)
dataset_train_C1 = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/mix_bottle/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/mix_bottle',
        pipeline=train_pipeline,
        classes = ('object',))  
dataset_train_C1_rp = dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_train_C1)
dataset_train_D1 = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/0628/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/0628',
        pipeline=train_pipeline,
        classes = ('object',)) 
dataset_train_D1_rp = dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_train_D1)        
dataset_train_E1 = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/0711/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/0711',
        pipeline=train_pipeline,
        classes = ('object',)) 
dataset_train_E1_rp = dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_train_E1)
dataset_train_F1 = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/0818/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/realdata/Desktop/bottle_cap/0818',
        pipeline=train_pipeline,
        classes = ('object',))  
dataset_train_F1_rp = dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_train_F1)               

dataset_val = dict(
        type=dataset_type,
        ann_file='/home/xjgao/Dataset/Autostore/realdata/Desktop/mix/0/dataset.json',
        img_prefix='/home/xjgao/Dataset/Autostore/realdata/Desktop/mix/0',
        pipeline=test_pipeline,
        classes = ('object',))

data = dict(
    train=[
        dataset_train_A, dataset_train_B, dataset_train_C, dataset_train_D, dataset_train_E,
        dataset_train_A1_rp, dataset_train_B1_rp, dataset_train_C1_rp, dataset_train_D1_rp, dataset_train_E1_rp, dataset_train_F1_rp,],
    val=dataset_val
    )