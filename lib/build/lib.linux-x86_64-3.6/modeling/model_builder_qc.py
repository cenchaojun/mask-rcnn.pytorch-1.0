from functools import wraps
import importlib
import logging

import nn as mynn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.nn.init as init

from core.config import cfg
from model.roi_layers import ROIPool, ROIAlign
import modeling.fast_rcnn_heads as fast_rcnn_heads
import utils.blob as blob_utils
import utils.resnet_weights_helper as resnet_utils
import pdb
import numpy as np

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        self.resolution = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.sampling_ratio = cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO

        # self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        # roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

        # self.fc1 = nn.Linear(2048 * roi_size ** 2, hidden_dim)  # orig: self.RPN.dim_out
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(  # e.g. modeling.ResNet.ResNet_roi_conv5_head
        #     self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)

        self.avgpool = nn.AvgPool2d(self.resolution)
        self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs_no_regression(2048)

        self._init_modules()
        # self._init_weights()

    def _init_modules(self):
        # pdb.set_trace()
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def forward(self, data, rois, labels_int32=None, roidb=None):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels_int32, roidb)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels_int32, roidb)

    def _forward(self, data, rois, labels_int32=None, roidb=None):
        # pdb.set_trace()
        im_data = data

        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)  # go through conv5 then 2 mlps
        device_id = blob_conv.get_device()
        num_images = data.shape[0]
        rois_final = torch.empty((0, 5)).cuda(device_id)

        # pdb.set_trace()

        for im_i in range(num_images):
            if self.training:
                im_i_boxes = rois[im_i]  # train
            else:
                im_i_boxes = rois  # test
            batch_inds = im_i * torch.ones((im_i_boxes.shape[0], 1)).cuda(device_id)
            im_i_rois = torch.cat((batch_inds, im_i_boxes), 1)
            rois_final = torch.cat((rois_final, im_i_rois), 0)

        box_feat = ROIAlign((self.resolution, self.resolution), self.Conv_Body.spatial_scale,  # 14, 1/32
                            self.sampling_ratio)(blob_conv, rois_final)  # (N*n, 2048, 7, 7)
        x = self.avgpool(box_feat)  # (N*n, 2048, 1, 1)

        # batch_size = box_feat.size(0)
        # x = F.relu(self.fc1(box_feat.view(batch_size, -1)), inplace=True)
        # x = F.relu(self.fc2(x), inplace=True)

        cls_score = self.Box_Outs(x)  # (N*n, 1), N:images, n:rois
        cls_score = cls_score.view(num_images, -1)  # (N, n)

        # pdb.set_trace()

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}

            # bbox only-classification loss
            loss_cls, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses_no_regression(cls_score, labels_int32)
            return_dict['losses']['loss_cls'] = loss_cls
            return_dict['metrics']['accuracy_cls'] = accuracy_cls

            # pytorch 0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else:
            # Testing
            return_dict['cls_score'] = cls_score

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = ROIPool((resolution, resolution), sc)(bl_in, rois)
                    elif method == 'RoIAlign':
                        xform_out = ROIAlign(
                            (resolution, resolution), sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = ROIPool((resolution, resolution), spatial_scale)(blobs_in, rois)
            elif method == 'RoIAlign':
                xform_out = ROIAlign(
                    (resolution, resolution), spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        # pdb.set_trace()
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if name == 'fc1' or name == 'fc2':  # LJY
                    continue
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
