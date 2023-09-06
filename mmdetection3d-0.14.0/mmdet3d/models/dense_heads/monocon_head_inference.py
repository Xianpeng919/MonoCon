import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.gaussian_target import (gaussian_radius, gen_gaussian_target)
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmdet3d.ops.attentive_norm import AttnBatchNorm2d

from .monocon_head import MonoConHead

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class MonoConHeadInference(MonoConHead):
    def __init__(self,
                 *args, **kwargs):
        super(MonoConHeadInference, self).__init__(*args, **kwargs)
        self.pred_bbox2d = False
        self.wh_head = None
        self.offset_head = None
        self.kpt_heatmap_head = None
        self.kpt_heatmap_offset_head = None

    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19
        for head in [self.center2kpt_offset_head, self.depth_head,
                     self.dim_head, self.dir_feat, self.dir_cls, self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward_single(self, feat):
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)

        center2kpt_offset_pred = self.center2kpt_offset_head(feat)
        dim_pred = self.dim_head(feat)

        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = 1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)
        return center_heatmap_pred, center2kpt_offset_pred, dim_pred, alpha_cls_pred, alpha_offset_pred, depth_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'center2kpt_offset_preds',
                          'dim_preds', 'alpha_cls_preds', 'alpha_offset_preds', 'depth_preds'))
    def loss(self,
             **kwargs):
        raise NotImplementedError

    def get_bboxes(self,
                   center_heatmap_preds,
                   center2kpt_offset_preds,
                   dim_preds,
                   alpha_cls_preds,
                   alpha_offset_preds,
                   depth_preds,
                   img_metas,
                   rescale=False):
        assert len(center_heatmap_preds) == len(center2kpt_offset_preds) \
               == len(dim_preds) == len(alpha_cls_preds) == len(alpha_offset_preds) == 1
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        box_type_3d = img_metas[0]['box_type_3d']

        batch_det_scores, batch_det_bboxes_3d, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            center2kpt_offset_preds[0],
            dim_preds[0],
            alpha_cls_preds[0],
            alpha_offset_preds[0],
            depth_preds[0],
            img_metas[0]['pad_shape'][:2],
            img_metas[0]['cam_intrinsic'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel,
            thresh=self.test_cfg.thresh)

        det_results = [
            [box_type_3d(batch_det_bboxes_3d,
                         box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)),
             batch_det_scores[:, -1],
             batch_labels,
             ]
        ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       center2kpt_offset_pred,
                       dim_pred,
                       alpha_cls_pred,
                       alpha_offset_pred,
                       depth_pred,
                       img_shape,
                       camera_intrinsic,
                       k=100,
                       kernel=3,
                       thresh=0.4):
        batch, cat, height, width = center_heatmap_pred.shape
        assert batch == 1
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, ys, xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        # decode 3D prediction
        dim = transpose_and_gather_feat(dim_pred, batch_index)
        alpha_cls = transpose_and_gather_feat(alpha_cls_pred, batch_index)
        alpha_offset = transpose_and_gather_feat(alpha_offset_pred, batch_index)
        depth_pred = transpose_and_gather_feat(depth_pred, batch_index)
        depth = depth_pred[:, :, 0:1]

        # change predict score based on sigma
        sigma = depth_pred[:, :, 1]
        sigma = torch.exp(-sigma)
        batch_scores *= sigma
        batch_scores = batch_scores[..., None]

        # 0. get kpts prediction
        center2kpt_offset = transpose_and_gather_feat(center2kpt_offset_pred, batch_index)
        center2kpt_offset = center2kpt_offset.view(batch, k, self.num_kpt * 2)[..., -2:]
        center2kpt_offset[..., ::2] += xs.view(batch, k, 1).expand(batch, k, 1)
        center2kpt_offset[..., 1::2] += ys.view(batch, k, 1).expand(batch, k, 1)

        kpts = center2kpt_offset

        kpts[..., ::2] *= (inp_w / width)
        kpts[..., 1::2] *= (inp_h / height)

        # 1. decode alpha
        alpha = self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)

        # 1.5 get projected center
        center2d = kpts  # (b, k, 2)

        # 2. recover rotY
        rot_y = self.recover_rotation(kpts, alpha, camera_intrinsic)  # (b, k, 3)

        # 2.5 recover box3d_center from center2d and depth
        center3d = torch.cat([center2d, depth], dim=-1).squeeze(0)
        center3d = self.pts2Dto3D(center3d, np.array(camera_intrinsic)).unsqueeze(0)

        # 3. compose 3D box
        batch_bboxes_3d = torch.cat([center3d, dim, rot_y], dim=-1)

        mask = batch_scores[..., -1] > thresh
        batch_scores = batch_scores[mask]
        batch_bboxes_3d = batch_bboxes_3d[mask]
        batch_topk_labels = batch_topk_labels[mask]

        return batch_scores, batch_bboxes_3d, batch_topk_labels
