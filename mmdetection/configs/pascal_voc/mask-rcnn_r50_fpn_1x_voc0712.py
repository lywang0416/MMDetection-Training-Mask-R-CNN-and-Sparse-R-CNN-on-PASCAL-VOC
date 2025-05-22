_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 数据集设置
dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'
dataset_meta = dict(
    classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
)
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# 训练数据加载器
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/voc07_trainval.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline={{_base_.train_pipeline}},
        metainfo=dict(classes=classes),
    )
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/voc07_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        metainfo=dict(classes=classes),
    )
)

# 测试数据加载器
test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/voc07_test.json',
    metric=['bbox', 'segm'],
    classwise=True,
    format_only=False,
)
test_evaluator = val_evaluator

# 优化器配置
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# 学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# 训练配置
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric=['bbox', 'segm'])

# 加载预训练权重
load_from = 'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'