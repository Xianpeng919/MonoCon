_base_ = [
    './monocon_dla34_200e_kitti.py'
]

model = dict(
    bbox_head=dict(
        type='MonoConHeadInference',
    )
)
