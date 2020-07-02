import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import pdb
import sys

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:  # bg and fg
            self.bbox_pred = nn.Linear(dim_in, 4 * 2)
        else:
            self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        # pdb.set_trace()
        cls_score = self.cls_score(x)
        # LJ always return cls_score without softmax
        # if not self.training:
        #     cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


class fast_rcnn_outputs_no_regression(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, 1)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = torch.sigmoid(cls_score)

        return cls_score


def fast_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)

    if cfg.FAST_RCNN.USE_CLS_WEIGHT:
        cls_weight = torch.ones(cls_score.size(1), dtype=torch.float32, device=device_id)
        cls_weight[0] = 0.5 # 原先 0.2
        loss_cls = F.cross_entropy(cls_score, rois_label, weight=cls_weight)
    else:
        loss_cls = F.cross_entropy(cls_score, rois_label)

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    # ForkedPdb().set_trace()

    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)

    return loss_cls, loss_bbox, accuracy_cls


def fast_rcnn_losses_no_regression(cls_score, label_int32):
    device_id = cls_score.get_device()
    # rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    rois_label = label_int32.float()

    if cfg.FAST_RCNN.USE_CLS_WEIGHT:
        cls_weight = torch.ones(cls_score.size(1), dtype=torch.float32, device=device_id)
        cls_weight[0] = 0.5 #
        loss_cls = F.cross_entropy(cls_score, rois_label, weight=cls_weight)
    else:
        # ForkedPdb().set_trace()
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, rois_label)

    cls_preds = (cls_score > 0).type_as(label_int32)
    accuracy_cls = cls_preds.eq(label_int32).float().mean()

    return loss_cls, accuracy_cls


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()
        if cfg.USE_CONTEXT_RELA or cfg.USE_INNER_CONTEXT_RELA:
            # 可学习矩阵
            self.matrix = torch.nn.Parameter(torch.randn((7, 7)))
        # 使用内部关系时 用于特征的维度改变
        # if cfg.USE_INNER_CONTEXT_RELA:
        #     self.fc3 = nn.Linear(dim_in, hidden_dim)
        #     mynn.init.XavierFill(self.fc3.weight)
        #     init.constant_(self.fc3.bias, 0)


    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret, part_rois):
        # 训练模式下 batch=2 测试模式下 batch=1
        # x=blobs_in [fpn5, fpn4, fpn3, fpn2]
        batch = x[1].shape[0]
        x = self.roi_xform(
            # add part_rois None
            x, rpn_ret, part_rois,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        # pdb.set_trace()
        batch_size = x.size(0)
        # 测试模式与训练模式不同 写错了
        if cfg.USE_CONTEXT_RELA:
            # 加上可学习矩阵 inplace=False
            # *= 原地操作 直接改变原值
            # pdb.set_trace()
            if self.training:
                x[-8:-6] *= self.matrix # -8:-6
                x[-3:-1] *= self.matrix # -3:-1
            else:
                x[-3:-1] *= self.matrix # -3:-1

        # if cfg.USE_INNER_CONTEXT_RELA:
        #     if self.training:
        #         x[-2 * 2 * cfg.INNER_ROI_SIZE:-1 * 2 * cfg.INNER_ROI_SIZE] *= self.matrix
        #         x[-1 * 2 * cfg.INNER_ROI_SIZE:] *= self.matrix
        #     else:
        #         x[-1 * 2 * cfg.INNER_ROI_SIZE:] *= self.matrix

        # if cfg.USE_INNER_CONTEXT_RELA:
        #     part_feat = x[-(5 * batch):]
        #     part_feat = part_feat.view(-1, part_feat.size(1))
        #     part_feat = F.relu(self.fc3(part_feat), inplace=True)
        #     # [490, 1024, 1, 1]
        #     part_feat = part_feat.view((part_feat.size(0), part_feat.size(1), 1, 1))
        #     # pdb.set_trace()

        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        # pdb.set_trace()

        # LJ 将part的特征提取出来 并改变维度
        if part_rois is not None:
            if cfg.USE_INNER_CONTEXT_RELA:
                box_feat = x[:-(cfg.INNER_ROI_SIZE[0] * cfg.INNER_ROI_SIZE[1] * 2 * batch)]
                part_feat = x[-(cfg.INNER_ROI_SIZE[0] * cfg.INNER_ROI_SIZE[1] * 2 * batch):]
            else:
                box_feat = x[:-(5 * batch)]
                part_feat = x[-(5 * batch):]

            # if not cfg.USE_INNER_CONTEXT_RELA:
            #     part_feat = x[-(5 * batch):]
            #     part_feat = part_feat.view((part_feat.size(0), part_feat.size(1), 1, 1))
            part_feat = part_feat.view((part_feat.size(0), part_feat.size(1), 1, 1))

        else:
            box_feat = x
            part_feat = None

        box_feat = box_feat.view((box_feat.size(0), box_feat.size(1), 1, 1))
        # pdb.set_trace()

        return box_feat, part_feat # x


class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*2): 'head_conv%d_w' % (i+1),
                'convs.%d.bias' % (i*2): 'head_conv%d_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*3): 'head_conv%d_w' % (i+1),
                'convs.%d.weight' % (i*3+1): 'head_conv%d_gn_s' % (i+1),
                'convs.%d.bias' % (i*3+1): 'head_conv%d_gn_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x
