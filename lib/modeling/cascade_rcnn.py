from modeling.generate_proposal_labels import GenerateProposalLabelsOp
import utils.boxes as box_utils
import utils.blob as blob_utils
from core.config import cfg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import modeling.fast_rcnn_heads as fast_rcnn_heads
from model.roi_layers import ROIPool, ROIAlign
from torch.autograd import Variable
import nn as mynn
import torch.nn.init as init
import roi_data.fast_rcnn
from datasets import json_dataset
import numpy.random as npr
from modeling.ResNet import *
import pdb

# 现在只支持c4模式
# 应返回numpy类型
def rois_refine(rois, bbox_pred, cls_score, im_info):
    device_id = bbox_pred.get_device()
    box_num1 = int(np.sum(rois[:, 0] == 0))
    box_num2 = int(rois.shape[0]) - box_num1
    temp_bbox_pred = [bbox_pred[:box_num1], bbox_pred[box_num1:]]
    temp_cls_score = [cls_score[:box_num1], cls_score[box_num1:]]
    temp_boxes = [rois[:box_num1], rois[box_num1:]]
    # box_num = [0, box_num1, box_num2 + box_num1]
    new_rois = np.zeros((box_num1 + box_num2, 5))
    # new_rois = torch.zeros((box_num1 + box_num2, 5)).cuda(device_id)
    box_num = [0, box_num1, box_num2 + box_num1]

    batch = len(im_info)
    im_info = im_info.data.numpy()
    for i in range(batch):
        im_scale = float(im_info[i][-1])
        im_shape = im_info[i][:2] / im_scale
        boxes = temp_boxes[i][:, 1:5] / im_scale
        # Apply bounding-box regression deltas
        box_deltas = temp_bbox_pred[i].data.cpu().numpy().squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        # Remove predictions for bg class (compat with MSRA code)
        # box_deltas = box_deltas[:, -4:]
        pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        # pred_boxes (512,14) 14组坐标差异不大
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im_shape)
        # pred_boxes = torch.from_numpy(pred_boxes).cuda(device_id)

        scores = F.softmax(temp_cls_score[i], dim=1)
        scores = scores.data.cpu().numpy().squeeze()
        # In case there is 1 proposal
        scores = scores.reshape([-1, scores.shape[-1]])
        # scores = scores.view([-1, scores.size(-1)])
        # -1 index为背景类
        max_index = np.argmax(scores, axis=1)
        # max_index = max_index[:, np.newaxis]
        # max_index = torch.argmax(scores, 1)
        # pdb.set_trace()
        new_boxes = np.zeros((pred_boxes.shape[0], 4))
        # new_boxes = torch.zeros((pred_boxes.size(0), 4)).cuda(device_id)
        # for j in range(pred_boxes.size(0)):
        # pdb.set_trace()
        for j in range(pred_boxes.shape[0]):
            new_boxes[j] = pred_boxes[j][max_index[j] * 4:(max_index[j] + 1) * 4]

        new_boxes = new_boxes * im_scale
        rois_index = np.zeros((pred_boxes.shape[0], 1))
        # rois_index = torch.zeros((pred_boxes.size(0), 1)).cuda(device_id)
        rois_index[:] = i
        new_rois[box_num[i] : box_num[i+1]] = np.append(rois_index, new_boxes, axis=1)
        # new_rois[box_num[i]: box_num[i + 1]] = torch.cat((rois_index, new_boxes), dim=1)
        # pdb.set_trace()
    new_rois = np.array(new_rois, dtype=np.float32)

    return new_rois

# class new_boxfeat(nn.Module):
#     def __init__(self, dim_in):
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
#         roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
#         self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#
#         self.resolution = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
#         self.sampling_ratio = cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
#
#         self._init_weights()
#
#     def _init_weights(self):
#         mynn.init.XavierFill(self.fc1.weight)
#         init.constant_(self.fc1.bias, 0)
#         mynn.init.XavierFill(self.fc2.weight)
#         init.constant_(self.fc2.bias, 0)
#
#     def forward(self, x, blob_conv, scale):
#         # x:rois
#         # x=blobs_in [fpn5, fpn4, fpn3, fpn2]
#         # fpn 有错误
#         if isinstance(blob_conv, list):
#             self.blobs_in = blob_conv[1]
#             self.spatial_scale = scale[1]
#         else:
#             self.blobs_in = blob_conv
#             self.spatial_scale = scale
#
#         self.device_id = self.blobs_in.get_device()
#         x = Variable(torch.from_numpy(x)).cuda(self.device_id)
#         # pdb.set_trace()
#         x = ROIAlign((self.resolution, self.resolution),
#                              self.spatial_scale, self.sampling_ratio)(self.blobs_in, x)
#
#         batch_size = x.size(0)
#         x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
#         x = F.relu(self.fc2(x), inplace=True)
#         box_feat = x
#         box_feat = box_feat.view((box_feat.size(0), box_feat.size(1), 1, 1))
#
#         return  box_feat

# def get_loss(cls_score, bbox_pred, rpn_ret):
#     # GenerateProposalLabels = GenerateProposalLabelsOp()
#     # # 生成box_target等 用于loss计算
#     # 正常cascade 会更新新的pos_box和neg_box 因此会用到新的IOU进行采样 三个stage 0.5,0.6,0.7
#     # return_dict = {}
#     # blobs_out = GenerateProposalLabels(rois, roidb, im_info)
#     # return_dict.update(blobs_out)
#
#     assert cls_score.size(0) == rpn_ret['labels_int32'].shape[0]
#     # pdb.set_trace()
#     loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
#         cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
#         rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
#
#     return loss_cls, loss_bbox, accuracy_cls


def add_fast_rcnn_blobs(blobs, im_scales, roidb, pos_iou):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = sample_rois(entry, im_scales[im_i], im_i, pos_iou)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

def sample_rois(roidb, im_scale, batch_idx, pos_iou):
    """Generate a random sample of RoIs comprising foreground and background examples.
    """
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))  # 0.25 x 512 by default
    max_overlaps = roidb['max_overlaps']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= pos_iou)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < pos_iou) &  # [0.0, 0.5) by default
                       (max_overlaps >= 0))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb['max_classes'][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb['boxes'][keep_inds]
    # pdb.set_trace()

    if 'bbox_targets' not in roidb:
        # pdb.set_trace()
        gt_inds = np.where(roidb['gt_classes'] > 0)[0]
        gt_boxes = roidb['boxes'][gt_inds, :]

        # print(gt_inds)
        # print(roidb['box_to_gt_ind_map'])
        # ForkedPdb().set_trace()

        if len(gt_inds) > 0:  # LJY
            # print(gt_inds)
            # print(roidb['box_to_gt_ind_map'])
            # print(roidb['box_to_gt_ind_map'][keep_inds])
            gt_assignments = gt_inds[roidb['box_to_gt_ind_map'][keep_inds]]
            bbox_targets = compute_targets(sampled_boxes, gt_boxes[gt_assignments, :], sampled_labels)
            bbox_targets, bbox_inside_weights = expand_bbox_targets(bbox_targets)
        else:  # all-negative image
            # generate dummy gt boxes
            gt_boxes = sampled_boxes.copy()
            # pdb.set_trace()
            bbox_targets = compute_targets(sampled_boxes, gt_boxes, sampled_labels)
            bbox_targets, bbox_inside_weights = expand_bbox_targets(bbox_targets)
            # pdb.set_trace()
    else:
        # LJ 不会进入
        pdb.set_trace()
        bbox_targets, bbox_inside_weights = expand_bbox_targets(roidb['bbox_targets'][keep_inds, :])

    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights)

    return blob_dict

def compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = box_utils.bbox_transform_inv(ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    # Use class "1" for all fg boxes if using class_agnostic_bbox_reg
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        labels.clip(max=1, out=labels)
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def get_loss(cls_score, bbox_pred, blobs):
    if cls_score.size(0) != blobs['labels_int32'].shape[0]:
        pdb.set_trace()
    loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
        cls_score, bbox_pred, blobs['labels_int32'], blobs['bbox_targets'],
        blobs['bbox_inside_weights'], blobs['bbox_outside_weights'])

    return loss_cls, loss_bbox, accuracy_cls

def get_new_blobs(rois, im_info, roidb, pos_iou=0):
    im_scales = im_info.data.numpy()[:, 2]
    output_blob_names = roi_data.fast_rcnn.get_fast_rcnn_blob_names()
    # pdb.set_trace()
    json_dataset.add_proposals(roidb, rois, im_scales, crowd_thresh=0)
    blobs = {k: [] for k in output_blob_names}
    add_fast_rcnn_blobs(blobs, im_scales, roidb, pos_iou)

    return blobs