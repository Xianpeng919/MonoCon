# dataset settings
dataset_type = 'KittiMonoDatasetMonoCon'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', to_float32=True, color_type='color'),
    dict(
        type='LoadAnnotations3DMonoCon',
        with_bbox=True,
        with_2D_kpts=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomShiftMonoCon', shift_ratio=0.5, max_shift_px=32),
    dict(type='RandomFlipMonoCon', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # Note: keys ['gt_kpts_2d', 'gt_kpts_valid_mask'] is hard coded in DefaultFormatBundle
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths', 'gt_kpts_2d', 'gt_kpts_valid_mask',
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'pad_shape', 'scale_factor', 'flip',
                   'cam_intrinsic', 'pcd_horizontal_flip',
                   'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'rect', 'Trv2c', 'P2', 'pcd_trans',
                   'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                   'pts_filename', 'transformation_3d_flow', 'cam_intrinsic_p0',)
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAugMonoCon',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlipMonoCon'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_train_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_train.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        min_height=25,
        min_depth=2,
        max_depth=65,
        max_truncation=0.5,
        max_occlusion=2,
        box_type_3d='Camera'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'))
evaluation = dict(interval=5)
