from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_layers import ROIPool, ROIAlign
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
import numpy as np
import pdb
import utils.boxes as box_utils
import core.test as core_test
import cv2
import math
import time
import os
from modeling.cascade_rcnn import *

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

#------------------------------
# LJ Code
#------------------------------
# 2019.9.20 参考源码 改
class Attention_Net(nn.Module):
    def __init__(self, dim_in):
        super(Attention_Net, self).__init__()
        # 无 fpn dim_in = 1024
        # note: rpn : the dim of class_weight, base_feature are 2048+1,1024
        # rpn + fpn: the dim of class_weight, base_feature are 1024+1,256
        if not cfg.FPN.FPN_ON:
            self.dim_out = dim_in // 16 # dim_in 1024
            self.fc = nn.Linear(self.dim_out, dim_in * 2 + 1)
        else:
            self.dim_out = dim_in // 4  # dim_in 256
            # fpn dim_in = 256 应转为 2014 与class_weight 匹配
            self.fc = nn.Linear(self.dim_out, dim_in * 4 + 1)

        self.conv = torch.nn.Conv2d(dim_in, self.dim_out, kernel_size=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.conv.bias, 0)
        torch.nn.init.normal_(self.fc.weight, std=0.01)
        torch.nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, class_weight): # D 1024
        out = self.conv(x) # [N,D,H,W] -> [N,D/16,H/2,W/2]
        out = self.avg_pool(out) # [N,D/16,H/2,W/2] -> [N,D/16,1,1]
        out = out.squeeze(3).squeeze(2) # [N,D/16,1,1] -> [N,D/16]
        out = self.fc(out) # [N,D/64] -> [N,D*2]
        out = self.relu(out) # 新加
        class_weight = class_weight.transpose(1, 0)
        # pdb.set_trace()
        out = torch.matmul(out, class_weight) # [N,C+1]
        out = F.softmax(out, dim=1) # softmax to raw

        return out

# 提取object与part的空间关系
# 对应公式（5）中的 (fgm, fgn)
def extract_position_matrix(boxes, part_boxes):
    # 参考 relation network 源码
    # 获取坐标
    xmin, ymin, xmax, ymax = torch.split(boxes, [1, 1, 1, 1], dim=1)
    xmin_part, ymin_part, xmax_part, ymax_part = torch.split(part_boxes, [1, 1, 1, 1], dim=1)
    # 获取box和part的中心 获取框的宽高
    box_width = xmax - xmin + 1
    box_height = ymax - ymin + 1
    part_width = xmax_part - xmin_part + 1
    part_height = ymax_part - ymin_part + 1
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    center_x_part = 0.5 * (xmin_part + xmax_part)
    center_y_part = 0.5 * (ymin_part + ymax_part)
    # log(|xm - xn| / wm) log(|ym - yn| / hm)
    center_x_part = torch.transpose(center_x_part, 1, 0)
    center_y_part = torch.transpose(center_y_part, 1, 0)
    delta_x = torch.abs((center_x - center_x_part) / box_width)
    delta_y = torch.abs((center_y - center_y_part) / box_height)
    delta_x[delta_x < 1e-3] = 1e-3
    delta_y[delta_y < 1e-3] = 1e-3
    delta_x = torch.log(delta_x)
    delta_y = torch.log(delta_y)
    # log(wn / wm) log(hn / hm)
    part_width = torch.transpose(part_width, 1, 0)
    part_height = torch.transpose(part_height, 1, 0)
    delta_width = part_width / box_width
    delta_height = part_height / box_height
    delta_width = torch.log(delta_width)
    delta_height = torch.log(delta_height)

    # 增加维度 并cat起来
    cat_list = [delta_x, delta_y, delta_width, delta_height]
    for i in range(len(cat_list)):
        cat_list[i] = cat_list[i].unsqueeze(-1)
    position_matrix = torch.cat((cat_list[0], cat_list[1]), dim=2)
    for i in range(2, len(cat_list)):
        position_matrix = torch.cat((position_matrix, cat_list[i]), dim=2)

    # position_matrix [512, 14, 4]
    return  position_matrix

# 位置矩阵编码
# 对应公式(5)中的 Eg(fgm, fgn) 升维编码
def extract_position_embedding(position_matrix, em_dim, wave_length=1000):
    feat_range = torch.arange(0, em_dim / 8) / (em_dim // 8) # float 类型
    feat_range = feat_range.cuda(torch.cuda.current_device())
    # 参考源码 dim_mat=[1., 2.37137365, 5.62341309,
    # # 13.33521461, 31.62277603, 74.98941803, 177.82794189, 421.69650269]
    dim_mat = torch.pow(wave_length, feat_range)
    # dim_mat [1, 1, 1, 8]
    dim_mat = dim_mat.view((1, 1, 1, -1))
    position_matrix = 100 * position_matrix
    # [512, 14, 4, 1]
    position_matrix = position_matrix.unsqueeze(-1)
    # div_mat [512, 14, 4, 8]
    div_mat = position_matrix / dim_mat
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding [512, 14, 4, 16]
    embedding = torch.cat((sin_mat, cos_mat), dim=3)
    # embedding [512, 14, 64]
    embedding = embedding.view((embedding.size(0), embedding.size(1), em_dim))

    return  embedding

# 提取part与relation的关系 提取方式可查阅论文
def extract_part_object_relation(box_feat, part_feat, position_embedding, Wg,
        Wq, Wk, conv_1x1, relu, group):
    # position_embedding [512, 14, 64]
    box_dim = position_embedding.size(0)
    part_dim = position_embedding.size(1)
    feat_dim = position_embedding.size(-1)
    # 前两维铺平 [7168, 64]
    position_embedding = position_embedding.view((box_dim * part_dim, feat_dim))
    # [7168, 16] fc_dim # 对应公式（5）中的可学矩阵Wg relu相当于max作用
    position_feat = relu(Wg(position_embedding))
    # [512, 14, 16]
    aff_weight = position_feat.view((box_dim, part_dim, group))
    # [512, 16, 14] 对应公式（5）计算后的 Wg
    aff_weight = torch.transpose(aff_weight, 2, 1)

    # query: box 部分的feature
    # [512, 2048, 1, 1] - > [512, 2048]
    box_feat = box_feat.squeeze(3).squeeze(2)
    # [512, 2048] - > [512, 1024] 对应公式（4）中的Wq（全连接层参数）
    q_data = Wq(box_feat)
    # 按group分组
    # [512, 1024] - > [512, 16, 64]
    q_data_batch = q_data.view((-1, group, q_data.size(-1) // group))
    # [512, 16, 64] - > [16, 512, 64]
    q_data_batch = torch.transpose(q_data_batch, 1, 0)

    # key: part 部分的feature 对应公式（4）中的Wk
    part_feat = part_feat.squeeze(3).squeeze(2)
    # [14, 2048] - > [14, 1024] 对应公式（4）中的Wk（全连接层参数）
    k_data = Wk(part_feat)
    # [14, 1024] - > [14, 16, 64]
    k_data_batch = k_data.view((-1, group, k_data.size(-1) // group))
    # [14, 16, 64] - > [16, 64, 14]
    k_data_batch = k_data_batch.transpose(1, 0).transpose(1, 2)
    # vaule: [14, 2048]
    v_data = part_feat
    # 对应公式（4）中的计算后的 Wa
    aff = torch.matmul(q_data_batch, k_data_batch)
    # 尺度变化
    aff_scale = (1.0 / group) * aff

    # 有个小bug 默认len(box)=512 但会存在box数量不够512的情况，导致part_box无返回值
    # 现已修改 ResNet.py 中的相关操作
    # if aff_scale.size(-1) == 0:
    #     pdb.set_trace()

    # [512, 16, 14] 对应公式（4）计算后的Wa
    aff_scale = aff_scale.transpose(1, 0)

    aff_weight[aff_weight < 1e-6] = 1e-6
    # 数学运算小技巧
    weighted_aff = torch.log(aff_weight) + aff_scale
    # pdb.set_trace()
    # 对应公式（3）中计算后的W
    aff_softmax = F.softmax(weighted_aff, dim=2)
    # pdb.set_trace()
    # [512 * 16, 14]
    aff_softmax = aff_softmax.view(-1, aff_softmax.size(-1))
    # [512 * 16, 14] dot [14, 2048] -> [512*16, 2048]
    output = torch.matmul(aff_softmax, v_data)
    # [512*16, 2048] -> [512, 16 * 2048, 1, 1]
    output = output.view((box_dim, -1, 1, 1))
    # 卷积层参数对应 Wv
    # [512, 16 * 2048, 1, 1] -> [512, 16 * 64, 1, 1]
    output = conv_1x1(output)

    return  output

# 将box可视化,查看结果是否正确
def vis_box(part_boxes, boxes, im_path, im_scale):
    vis_dir = '/home/lianjie/mask-rcnn.pytorch-1.0/box_vis'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    img_name = im_path.split('/')[-1]
    img = cv2.imread(im_path)
    part_boxes = part_boxes / im_scale
    boxes = boxes / im_scale
    for i in range(len(part_boxes)):
        point0 = tuple(part_boxes[i][:2])
        point1 = tuple(part_boxes[i][2:])
        img = cv2.rectangle(img, point0, point1, (0, 255, 0), 2)
    for i in range(len(boxes)):
        point0 = tuple(boxes[i][:2])
        point1 = tuple(boxes[i][2:])
        img = cv2.rectangle(img, point0, point1, (255, 0, 0), 2)

    save_path = os.path.join(vis_dir, img_name)
    cv2.imwrite(save_path, img)
    # pdb.set_trace()

# 合并part_boxes和parts_list 转换为part_rois 送入roi pooling
def combine_part_info(part_boxes, parts_list, im_info, index):
    # 增加维度 [14, 1]
    parts_list = parts_list[:, np.newaxis]
    # 一定注意 box[index,x1,y1,x2,y2] 注意：index 对应batch的index
    parts_list[:, 0] = index
    im_scale = float(im_info[index][-1])
    # 在测试模式下 也有scale操作
    part_boxes = part_boxes * im_scale
    # [14, 5]
    part_rois = np.append(parts_list, part_boxes, axis=1)
    part_rois = np.array(part_rois, dtype=np.float32)

    return part_rois, part_boxes

# Resoning network 论文 类别关系
def get_cls_relation(att_net, blob_conv, cls_weight, adj_matrix, fc,
                     relu, cls_score_old, current_device):

    cls_att = att_net(blob_conv, cls_weight)
    # [1, 31] -> [31, 1]
    cls_att = cls_att.view((cls_att.size(1), 1))
    adj_matrix = adj_matrix.cuda(current_device)
    temp1 = torch.matmul(adj_matrix, cls_weight)  # EM
    temp2 = cls_att * temp1  # a x EM
    temp3 = relu(fc(temp2))  # 降维
    cls_p = F.softmax(cls_score_old, dim=1)
    cls_relation = torch.matmul(cls_p, temp3)  # P(a x EM)
    cls_relation = cls_relation.view((cls_relation.size(0), cls_relation.size(1), 1, 1))

    return cls_relation

# relation network 论文 object part 关系
def get_spa_relation(boxes, part_boxes, em_dim, current_device):
    boxes = torch.from_numpy(boxes).cuda(current_device).detach()
    part_boxes = torch.from_numpy(part_boxes).cuda(current_device)
    position_matrix = extract_position_matrix(boxes, part_boxes)
    position_embedding = extract_position_embedding(position_matrix, em_dim)
    spa_relation = position_embedding.view((position_embedding.size(0), -1, 1, 1))

    return spa_relation

# relation network 论文 object part 关系
def get_context_relation(boxes, lung_boxes, current_device, em_dim, box_feat, lung_feat,
                             Wg, Wq, Wk, conv_1x1, relu_part, group):
    boxes = torch.from_numpy(boxes).cuda(current_device).detach()
    lung_boxes = torch.from_numpy(lung_boxes).cuda(current_device)
    position_matrix = extract_position_matrix(boxes, lung_boxes)
    position_embedding = extract_position_embedding(position_matrix, em_dim)
    # part_object_relation = position_embedding.view((position_embedding.size(0), -1, 1, 1))
    context_relation = extract_part_object_relation(box_feat, lung_feat, position_embedding,
         Wg, Wq, Wk, conv_1x1, relu_part, group)

    return context_relation

# 将肺野区域的box坐标分解为x组
def split_boxes(part_boxes, roi_size):
    rois_num = part_boxes.shape[0]
    boxes = np.zeros((roi_size * rois_num, 4), dtype=np.float32)
    roi_size_sqrt = int(math.sqrt(roi_size))

    for i in range(rois_num):
        roi_index = i * roi_size
        deta_x = (part_boxes[i][2] - part_boxes[i][0]) / roi_size_sqrt
        deta_y = (part_boxes[i][3] - part_boxes[i][1]) / roi_size_sqrt
        # 保证转换后的 左上 右下 坐标不失真
        boxes[roi_index][0] = part_boxes[i][0]
        boxes[roi_index][1] = part_boxes[i][1]
        boxes[roi_index][2] = part_boxes[i][0] + deta_x
        boxes[roi_index][3] = part_boxes[i][1] + deta_y

        for j in range(1, roi_size):
            boxes[roi_index + j][0] = boxes[roi_index + j - 1][0] + deta_x
            boxes[roi_index + j][1] = boxes[roi_index + j - 1][1]
            boxes[roi_index + j][2] = boxes[roi_index + j - 1][2] + deta_x
            boxes[roi_index + j][3] = boxes[roi_index + j - 1][3]
            if j % roi_size_sqrt == 0:
                row_index = j // roi_size_sqrt
                boxes[roi_index + j][0] = boxes[(row_index - 1) * roi_size_sqrt + roi_index][0]
                boxes[roi_index + j][1] = boxes[(row_index - 1) * roi_size_sqrt + roi_index][1] + deta_y
                boxes[roi_index + j][2] = boxes[(row_index - 1) * roi_size_sqrt + roi_index][2]
                boxes[roi_index + j][3] = boxes[(row_index - 1) * roi_size_sqrt + roi_index][3] + deta_y
            # 保证转换后的 左上 右下 坐标不失真
            if boxes[roi_index + j][2] > part_boxes[i][2]:
                boxes[roi_index + j][2] = part_boxes[i][2]
            if boxes[roi_index + j][3] > part_boxes[i][3]:
                boxes[roi_index + j][3] = part_boxes[i][3]

    # pdb.set_trace()
    return boxes

# 将肺野区域的box坐标非均匀分解为x组
def split_boxes2(part_boxes, roi_size):
    rois_num = part_boxes.shape[0]
    boxes = np.zeros((roi_size[0] * roi_size[1] * rois_num, 4), dtype=np.float32)
    # [3x1] 三行一列
    roi_row = roi_size[0]
    roi_col = roi_size[1]

    for i in range(rois_num):
        roi_index = i * roi_size[0] * roi_size[1]
        deta_x = (part_boxes[i][2] - part_boxes[i][0]) / roi_col
        deta_y = (part_boxes[i][3] - part_boxes[i][1]) / roi_row
        # 保证转换后的 左上 右下 坐标不失真
        boxes[roi_index][0] = part_boxes[i][0]
        boxes[roi_index][1] = part_boxes[i][1]
        boxes[roi_index][2] = part_boxes[i][0] + deta_x
        boxes[roi_index][3] = part_boxes[i][1] + deta_y

        for j in range(1, roi_size[0] * roi_size[1]):
            boxes[roi_index + j][0] = boxes[roi_index + j - 1][0] + deta_x
            boxes[roi_index + j][1] = boxes[roi_index + j - 1][1]
            boxes[roi_index + j][2] = boxes[roi_index + j - 1][2] + deta_x
            boxes[roi_index + j][3] = boxes[roi_index + j - 1][3]
            if j % roi_col == 0:
                row_index = j // roi_col
                boxes[roi_index + j][0] = boxes[(row_index - 1) * roi_col + roi_index][0]
                boxes[roi_index + j][1] = boxes[(row_index - 1) * roi_col + roi_index][1] + deta_y
                boxes[roi_index + j][2] = boxes[(row_index - 1) * roi_col + roi_index][2]
                boxes[roi_index + j][3] = boxes[(row_index - 1) * roi_col + roi_index][3] + deta_y
            # 保证转换后的 左上 右下 坐标不失真
            if boxes[roi_index + j][2] > part_boxes[i][2]:
                boxes[roi_index + j][2] = part_boxes[i][2]
            if boxes[roi_index + j][3] > part_boxes[i][3]:
                boxes[roi_index + j][3] = part_boxes[i][3]

    # pdb.set_trace()
    return boxes

class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        self._init_modules()

        # 参考reasoning network源码
        self.att_net = Attention_Net(self.Conv_Body.dim_out)
        graph_path = '/home/lianjie/mask-rcnn.pytorch-1.0/graph/graph_gk_yy.npy'
        self.adj_matrix = np.load(graph_path)
        self.adj_matrix = torch.from_numpy(self.adj_matrix)
        self.adj_matrix = self.adj_matrix.type(torch.FloatTensor)
        # 增强的特征维度
        self.cls_rela_dim = 256
        # 降维
        self.graph_weight_fc = nn.Linear(self.Box_Head.dim_out + 1, self.cls_rela_dim)
        self.relu_cls = nn.ReLU(inplace=True)

        # 维度对应relation network 源码
        # em_dim group 相互独立 互不影响
        # 空间关系编码维度
        self.em_dim = 64 # 64
        # 论文里分组进行卷积（待理解）
        self.group = 16
        # Wg, Wq, Wk 对应论文
        self.Wg = nn.Linear(self.em_dim, self.group)
        # inplace 应设为 False (原因未知)
        self.relu_part = nn.ReLU(inplace=False)
        # 增强后的特征维数
        self.context_rela_dim = self.group * self.em_dim # 1024
        # self.context_rela_dim = self.group * 32  # 512
        # box_feat
        self.Wq = nn.Linear(self.Box_Head.dim_out, self.context_rela_dim)
        # part_feat  part_feat和box_feat 会通过一样的roi pooling 维度一样
        self.Wk = nn.Linear(self.Box_Head.dim_out, self.context_rela_dim)
        # self.conv_1x1 = torch.nn.Conv2d(self.group * self.Box_Head.dim_out,
        #     self.group * self.em_dim, kernel_size=1, stride=1, groups=self.group)
        self.conv_1x1 = torch.nn.Conv2d(self.group * self.Box_Head.dim_out,
            self.context_rela_dim, kernel_size=1, stride=1, groups=self.group)
        torch.nn.init.normal_(self.conv_1x1.weight, mean=0.0, std=0.02)

        self.part_num = 5
        self.spa_rela_dim = self.em_dim * self.part_num

        self.norm_cls = torch.nn.LayerNorm(self.cls_rela_dim)
        self.norm_spa = torch.nn.LayerNorm(self.spa_rela_dim)
        self.norm_context = torch.nn.LayerNorm(self.context_rela_dim)
        self.relu = nn.ReLU(inplace=True)

        if cfg.USE_INNER_CONTEXT_RELA:
            self.inner_context_rela_dim = self.context_rela_dim  # 1024
            self.roi_size = cfg.INNER_ROI_SIZE # 3 ** 2

        if not cfg.USE_CLS_RELA and not cfg.USE_SPA_RELA and not cfg.USE_CONTEXT_RELA:
            self.feat_dim = self.Box_Head.dim_out
        elif cfg.USE_CLS_RELA and not cfg.USE_SPA_RELA and not cfg.USE_CONTEXT_RELA:
            self.feat_dim = self.Box_Head.dim_out + self.cls_rela_dim
        elif not cfg.USE_CLS_RELA and cfg.USE_SPA_RELA and not cfg.USE_CONTEXT_RELA:
            self.feat_dim = self.Box_Head.dim_out + self.spa_rela_dim
        elif not cfg.USE_CLS_RELA and not cfg.USE_SPA_RELA and cfg.USE_CONTEXT_RELA:
            self.feat_dim = self.Box_Head.dim_out + self.context_rela_dim
        elif not cfg.USE_CLS_RELA and cfg.USE_SPA_RELA and cfg.USE_CONTEXT_RELA:
            self.feat_dim = self.Box_Head.dim_out  + self.spa_rela_dim + self.context_rela_dim
        elif cfg.USE_CLS_RELA and cfg.USE_SPA_RELA and cfg.USE_CONTEXT_RELA:
            self.feat_dim = self.Box_Head.dim_out + self.cls_rela_dim + self.spa_rela_dim + self.context_rela_dim
        # self.feat_dim = self.Box_Head.dim_out * 3
        # if cfg.USE_INNER_CONTEXT_RELA:
        #     self.feat_dim = self.Box_Head.dim_out + self.inner_context_rela_dim
        self.Box_Outs_New = fast_rcnn_heads.fast_rcnn_outputs(self.feat_dim)

        if cfg.USE_CASCADE:
            self.Box_Outs_1 = fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)
            self.Box_Outs_2 = fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)
            # self.get_new_boxfeat = new_boxfeat(self.RPN.dim_out)


    def _init_modules(self):
        # pdb.set_trace()
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
            if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Keypoint_Head.res5.state_dict(), self.Box_Head.res5.state_dict())

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, im_info, im=None, roidb=None, **rpn_kwargs): # add im
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, im, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, im, roidb, **rpn_kwargs)

    # LJ 增加测试模式下的rodib读取
    def _forward(self, data, im_info, im=None, roidb=None, **rpn_kwargs): # add im (test) and roidb['image'] (train)
        # 已经 scale 后的图像 测试模式下也会scale
        im_data = data                                 # provide im_shape for box transfer see: test.py _get_blobs()
        # pdb.set_trace()                              # and rpn.py add_rpn_blobs()
        # 增加了测试模式下的roidb 不需要转换形式 直接用即可
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        # pdb.set_trace()

        # LJ 不管什么模式 总是返回标准形态的roidb roidb下的坐标均没有scale im_data为已经scale的数据
        # roidb 数据读取方式：
        # 训练：读取数据后，进行翻转操作，过滤掉一部分数据，再生成不同的scale，改变数据尺度
        #      此时roidb中的box均为标准尺度，在实际使用时，才会进行scale
        # 读取parts的坐标 训练模式和测试模式读取的roidb区别 最早版本读取错误 现已改正
        # 普通的rpn 模式
        if cfg.USE_SPA_RELA or cfg.USE_CONTEXT_RELA or cfg.USE_INNER_CONTEXT_RELA or cfg.USE_LUNG_FEATURE:
            if not cfg.FPN.FPN_ON:
                if self.training:
                    part_boxes = roidb[0]['part_boxes']
                    parts_list = roidb[0]['parts_list']
                else:
                    part_boxes = roidb['part_boxes']
                    parts_list = roidb['parts_list']

                # batch = 1, index = 0
                index = 0
                # 返回 part_rois 和 已经scale的part_boxes
                part_rois, part_boxes = combine_part_info(part_boxes, parts_list, im_info, index)

            # rpn+fpn 模式
            else:
                if self.training:
                    # part 数量5
                    part_num = self.part_num
                    # 训练模式batch=2 测试模式下 batch=1
                    batch_size = len(roidb)
                    part_rois = np.zeros((part_num * batch_size, 5), dtype=np.float32)
                    part_boxes = np.zeros((part_num * batch_size, 4), dtype=np.float32)
                    for i in range(batch_size):
                        temp_part_boxes = roidb[i]['part_boxes']
                        temp_parts_list = roidb[i]['parts_list']
                        # 第一次改的有错误 part_boxes 没有进行scale
                        temp_part_rois, temp_part_boxes = combine_part_info(temp_part_boxes, temp_parts_list, im_info, i)
                        part_rois[part_num * i : part_num * (i + 1)] = temp_part_rois
                        part_boxes[part_num * i : part_num * (i + 1)] = temp_part_boxes
                else:
                    # 测试模式batch=1
                    part_boxes = roidb['part_boxes']
                    parts_list = roidb['parts_list']
                    part_rois, part_boxes = combine_part_info(part_boxes, parts_list, im_info, 0)
        else:
            part_rois = None
            part_boxes = None

        if cfg.USE_INNER_CONTEXT_RELA:
            # pdb.set_trace()
            if self.training:
                batch_size = len(roidb)
            else:
                batch_size = 1
            part_num = self.part_num
            roi_size = self.roi_size[0] * self.roi_size[1]
            # 第一次操作 赋值错误
            temp_lung_boxes = [part_boxes[:part_num][2:4], part_boxes[part_num:][2:4]]
            split_part_rois = np.zeros((roi_size * 2 * batch_size, 5), dtype=np.float32)
            split_part_boxes = np.zeros((roi_size * 2 * batch_size, 4), dtype=np.float32)
            # pdb.set_trace()
            for i in range(batch_size):
            # 将肺部区域分离出来并均分为3×3
                lung_boxes = split_boxes2(temp_lung_boxes[i], self.roi_size)
                lung_index = np.ones((roi_size * 2 , 1)) * i
                lung_rois = np.append(lung_index, lung_boxes, axis=1)
                split_part_rois[i * roi_size * 2 : (i + 1) * roi_size * 2] = lung_rois
                split_part_boxes[i * roi_size * 2 : (i + 1) * roi_size * 2] = lung_boxes

            # pdb.set_trace()
            part_boxes = split_part_boxes
            part_rois = split_part_rois

        # pdb.set_trace()

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        # rpn 直接返回resnet conv4的feature 1024维 roi pooling 后再过一遍conv5 变为2048维
        # rpn+fpn 返回resnet conv2-conv5的feature 通过跨层1x1操作 均为256维 还有一层额外操作生成'fpn6'
        #   维度通过fc最终变为1024维 此时len(blob_conv) = 5
        blob_conv = self.Conv_Body(im_data)  # (2, 1024, 50, 50)

        # LJ rpn_ret 关键字 rois 返回roi区域的坐标 所有rois坐标为scale的版本
        # 在FPN模式下，分层做roi proposal  loss也会有所区别
        # 训练模式下 才会用到roidb
        # if len(roidb[0]['boxes']) == 0:
        #     pdb.set_trace()
        rpn_ret = self.RPN(blob_conv, im_info, roidb)  # rpn_ret: dict{}
        # pdb.set_trace()

        # 统计rois的index字段 index代表batch index=0,1
        # 获得每张图片的box数量 因为提出的框数量可能不够 512 需要统计具体数量
        if cfg.FPN.FPN_ON:
            # rpn_ret['rois'] -> [N*Nr, 5]
            box_num1 = int(np.sum(rpn_ret['rois'][:, 0] == 0))
            box_num2 = int(rpn_ret['rois'].shape[0]) - box_num1

        # if self.training:
        #     # can be used to infer fg/bg ratio
        #     return_dict['rois_label'] = rpn_ret['labels_int32']

        # 只取conv2-conv5
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            # 即 fpn6 在roi pooling 操作中用到fpn2-5 rpn操作中用到fpn2-fpn6
            # fpn6 通过 fpn5 stride=2 的降采样得到 此时 len(blob_conv) = 4
            blob_conv = blob_conv[-self.num_roi_levels:]

        # pdb.set_trace()

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.SHARE_RES5 and self.training:
                # rpn:训练模式进入
                box_feat, res5_feat, part_feat = self.Box_Head(blob_conv, rpn_ret, part_rois)  # box_feat: (N*512, 2048, 1, 1)
                # box_feat, res5_feat, _ = self.Box_Head(blob_conv, rpn_ret, None)
                # pdb.set_trace()
            else:
                # rpn:测试模式进入  rpn+fpn: 训练测试均进入
                # rpn: conv4的feature 1024维 roi pooling 后再过一遍conv5 变为2048维 输出[N*512, 2048]
                # rpn+fpn: rpn_ret 每个尺度负责256通道的roi pooling 最后通过fc变成1024 输出[N*512, 1024]
                box_feat, part_feat = self.Box_Head(blob_conv, rpn_ret, part_rois)
                # box_feat, _ = self.Box_Head(blob_conv, rpn_ret, None)
                # pdb.set_trace()

            if not cfg.USE_CASCADE:
                if cfg.USE_CLS_RELA or cfg.USE_SPA_RELA or cfg.USE_CONTEXT_RELA:
                    cls_score_old, bbox_pred_old = self.Box_Outs(box_feat)  # cls_score: (N*512, C)

            # pdb.set_trace()

            # 提取part与object的关系
            rois = rpn_ret['rois']
            # 从rpn 网络出来 为scale版本的boxes 应与part_boxes在同尺度
            boxes = rois[:, 1:5]
            # 现只支持c4模式
            if cfg.USE_CASCADE:
                spatial_scale = self.Conv_Body.spatial_scale
                # pdb.set_trace()
                # cascade stage 1
                cls_score_1, bbox_pred_1 = self.Box_Outs_1(box_feat)
                with torch.no_grad():
                    new_rois_1 = rois_refine(rois, bbox_pred_1, cls_score_1, im_info)
                if self.training:
                    loss_cls_1, loss_bbox_1, _ = get_loss(cls_score_1, bbox_pred_1, rpn_ret)
                    blobs_1 = get_new_blobs(new_rois_1, im_info, roidb, pos_iou=0.6)
                    new_rois_1 = blobs_1['rois']
                # with torch.no_grad():
                box_feat_1 = self.Box_Head(blob_conv, None, None, cascade=True, rois=new_rois_1)

                # cascade stage 2
                cls_score_2, bbox_pred_2 = self.Box_Outs_2(box_feat_1)
                with torch.no_grad():
                    new_rois_2 = rois_refine(new_rois_1, bbox_pred_2, cls_score_2, im_info)
                if self.training:
                    loss_cls_2, loss_bbox_2, _ = get_loss(cls_score_2, bbox_pred_2, blobs_1)
                    blobs_2 = get_new_blobs(new_rois_2, im_info, roidb, pos_iou=0.7)
                    new_rois_2 = blobs_2['rois']
                # with torch.no_grad():
                box_feat_2 = self.Box_Head(blob_conv, None, None, cascade=True, rois=new_rois_2)

                # cascade stage 2
                if self.training:
                    loss_cls_cascade = loss_cls_1 + loss_cls_2
                    loss_bbox_cascade = loss_bbox_1 + loss_bbox_2
                # 更新一些必要参数
                box_feat = box_feat_2
                if cfg.USE_CLS_RELA or cfg.USE_SPA_RELA or cfg.USE_CONTEXT_RELA:
                    boxes = new_rois_2[:, 1:5]
                    cls_score_old, bbox_pred_old = self.Box_Outs(box_feat)  # cls_score: (N*512, C)
                # c4模式用不到 box_num
                # box_num1 = int(np.sum(new_rois_2[:, 0] == 0))
                # box_num2 = int(new_rois_2.shape[0]) - box_num1

                # pdb.set_trace()

            # LJ
            # ----------------
            # 源码
            # ----------------
            # detach 截断反向梯度流 旧box的梯度更新通过loss途径(与自己版本的差别)
            current_device = torch.cuda.current_device()
            if cfg.USE_CLS_RELA:
                # 类别关系 [31, D] D: rpn 2048 rpn+fpn 1024
                global_semantic_pool = torch.cat((self.Box_Outs.cls_score.weight,
                                                  self.Box_Outs.cls_score.bias.unsqueeze(1)), 1).detach() # detach 很关键

                if not cfg.FPN.FPN_ON:
                    cls_relation = get_cls_relation(self.att_net, blob_conv,
                        global_semantic_pool, self.adj_matrix, self.graph_weight_fc,
                        self.relu_cls, cls_score_old, current_device)
                else:
                    # 使用 fpn4 的base feature
                    # blob_conv() [fpn5, fpn4, pfn3, fpn2]
                    batch_size = blob_conv[1].size(0)
                    # 测试模式 batch=1 box_num2=0 并不冲突
                    temp_cls_score = [cls_score_old[:box_num1], cls_score_old[box_num1 : (box_num2 + box_num1)]]
                    # pdb.set_trace()
                    cls_relation = torch.zeros((box_num1 + box_num2, self.cls_rela_dim, 1, 1)).cuda(current_device)
                    box_num = [0, box_num1, box_num2 + box_num1]
                    # pdb.set_trace()
                    for i in range(batch_size):
                        # 采用 fpn4 的base feature
                        temp_blob_conv = blob_conv[1][i].unsqueeze(0)
                        # pdb.set_trace()
                        temp_cls_relation = get_cls_relation(self.att_net, temp_blob_conv,
                            global_semantic_pool, self.adj_matrix, self.graph_weight_fc,
                            self.relu_cls, temp_cls_score[i], current_device)
                        # pdb.set_trace()
                        cls_relation[box_num[i] : box_num[i + 1]] = temp_cls_relation

                cls_relation = cls_relation.squeeze(3).squeeze(2)
                cls_relation = self.relu(self.norm_cls(cls_relation))
                cls_relation = cls_relation.unsqueeze(2).unsqueeze(3)

            # pdb.set_trace()
            # if self.training:
            #     im_path = roidb[0]['image']
            # else:
            #     im_path = roidb['image']
            # vis_box(part_boxes, boxes, im_path, im_scale)

            # part_object relation
            if cfg.USE_SPA_RELA:
                if not cfg.FPN.FPN_ON:
                    spa_relation = get_spa_relation(boxes, part_boxes, self.em_dim, current_device)
                else:
                    part_num = self.part_num
                    temp_boxes = [boxes[:box_num1], boxes[box_num1 : (box_num2 + box_num1)]]
                    temp_part_boxes = [part_boxes[:part_num], part_boxes[part_num : part_num * 2]]
                    spa_relation = torch.zeros((box_num1 + box_num2, self.spa_rela_dim, 1, 1)).cuda(current_device)
                    box_num = [0, box_num1, box_num2 + box_num1]
                    batch_size = blob_conv[1].size(0)
                    for i in range(batch_size):
                        temp_spa_relation = get_spa_relation(temp_boxes[i], temp_part_boxes[i],
                                self.em_dim, current_device)
                        spa_relation[box_num[i]: box_num[i + 1]] = temp_spa_relation

                spa_relation = spa_relation.squeeze(3).squeeze(2)
                spa_relation = self.relu(self.norm_spa(spa_relation))
                spa_relation = spa_relation.unsqueeze(2).unsqueeze(3)

            if cfg.USE_CONTEXT_RELA:
                if not cfg.FPN.FPN_ON:
                    # 左肺 右肺
                    # rebuttal 加上心影
                    lung_boxes = part_boxes[2:4] # 2:4
                    lung_feat = part_feat[2:4]
                    context_relation = get_context_relation(boxes, lung_boxes, current_device,
                        self.em_dim, box_feat, lung_feat,
                        self.Wg, self.Wq, self.Wk, self.conv_1x1, self.relu_part, self.group)
                else:
                    part_num = self.part_num
                    temp_boxes = [boxes[:box_num1], boxes[box_num1 : (box_num2 + box_num1)]]
                    temp_lung_boxes = [part_boxes[:part_num][2:4], part_boxes[part_num : part_num * 2][2:4]]
                    temp_box_feat = [box_feat[:box_num1], box_feat[box_num1 : (box_num2 + box_num1)]]
                    temp_lung_feat = [part_feat[:part_num][2:4], part_feat[part_num : part_num * 2][2:4]]
                    context_relation = torch.zeros((box_num1 + box_num2, self.context_rela_dim, 1, 1)).cuda(current_device)
                    box_num = [0, box_num1, box_num2 + box_num1]
                    batch_size = blob_conv[1].size(0)
                    # pdb.set_trace()
                    for i in range(batch_size):
                        temp_context_relation = get_context_relation(temp_boxes[i], temp_lung_boxes[i],
                                current_device, self.em_dim, temp_box_feat[i], temp_lung_feat[i],
                                self.Wg, self.Wq, self.Wk, self.conv_1x1, self.relu_part, self.group)
                        context_relation[box_num[i] : box_num[i + 1]] = temp_context_relation

                context_relation = context_relation.squeeze(3).squeeze(2)
                context_relation = self.relu(self.norm_context(context_relation))
                context_relation = context_relation.unsqueeze(2).unsqueeze(3)

            if cfg.USE_INNER_CONTEXT_RELA:
                roi_size =  self.roi_size[0] * self.roi_size[1]
                if not cfg.FPN.FPN_ON:
                    inter_context_relation = get_context_relation(boxes, part_boxes, current_device,
                        self.em_dim, box_feat, part_feat,
                        self.Wg, self.Wq, self.Wk, self.conv_1x1, self.relu_part, self.group)
                else:
                    temp_boxes = [boxes[:box_num1], boxes[box_num1: (box_num2 + box_num1)]]
                    temp_lung_boxes = [part_boxes[:roi_size * 2], part_boxes[roi_size * 2:]]
                    temp_box_feat = [box_feat[:box_num1], box_feat[box_num1: (box_num2 + box_num1)]]
                    temp_lung_feat = [part_feat[:2 * roi_size], part_feat[2 * roi_size:]]
                    inner_context_relation = torch.zeros((box_num1 + box_num2, self.inner_context_rela_dim, 1, 1)).cuda(current_device)
                    box_num = [0, box_num1, box_num2 + box_num1]
                    batch_size = blob_conv[1].size(0)
                    # pdb.set_trace()
                    for i in range(batch_size):
                        # pdb.set_trace()
                        temp_inner_context_relation = get_context_relation(temp_boxes[i], temp_lung_boxes[i],
                                current_device, self.em_dim, temp_box_feat[i], temp_lung_feat[i],
                                self.Wg, self.Wq, self.Wk, self.conv_1x1, self.relu_part, self.group)
                        inner_context_relation[box_num[i]: box_num[i + 1]] = temp_inner_context_relation

            if cfg.USE_LUNG_FEATURE:
                if not cfg.FPN.FPN_ON:
                    lung_feat = part_feat[2:4]
                    lung_feat = lung_feat.view((1, -1, 1, 1))
                    lung_relation = lung_feat.repeat(box_feat.size(0), 1, 1, 1)
                else:
                    part_num = self.part_num
                    temp_lung_feat = [part_feat[:part_num][2:4], part_feat[part_num:][2:4]]
                    lung_relation = torch.zeros((box_num1 + box_num2, box_feat.size(1)*2, 1, 1)).cuda(current_device)
                    box_num = [0, box_num1, box_num2 + box_num1]
                    repeat_num = [box_num1, box_num2]
                    batch_size = blob_conv[1].size(0)
                    for i in range(batch_size):
                        lung_feat = temp_lung_feat[i]
                        lung_feat = lung_feat.view((1, -1, 1, 1))
                        lung_feat = lung_feat.repeat(repeat_num[i], 1, 1, 1)
                        lung_relation[box_num[i]: box_num[i + 1]] = lung_feat

            # pdb.set_trace()
            if not cfg.USE_CLS_RELA and not cfg.USE_SPA_RELA and not cfg.USE_CONTEXT_RELA:
                f = box_feat
            elif cfg.USE_CLS_RELA and not cfg.USE_SPA_RELA and not cfg.USE_CONTEXT_RELA:
                f = torch.cat((box_feat, cls_relation), dim=1)
            elif not cfg.USE_CLS_RELA and cfg.USE_SPA_RELA and not cfg.USE_CONTEXT_RELA:
                f = torch.cat((box_feat, spa_relation), dim=1)
            elif not cfg.USE_CLS_RELA and not cfg.USE_SPA_RELA and cfg.USE_CONTEXT_RELA:
                f =  torch.cat((box_feat, context_relation), dim=1)
            elif not cfg.USE_CLS_RELA and cfg.USE_SPA_RELA and cfg.USE_CONTEXT_RELA:
                f1 =  torch.cat((spa_relation, context_relation), dim=1)
                f = torch.cat((box_feat, f1), dim=1)
            elif cfg.USE_CLS_RELA and cfg.USE_SPA_RELA and cfg.USE_CONTEXT_RELA:
                # gk_yy
                f1 = torch.cat((cls_relation, spa_relation), dim=1)
                f2 = torch.cat((f1, context_relation), dim=1)
                f = torch.cat((box_feat, f2), dim=1)
                # all_yy
                # f1 = torch.cat((spa_relation, context_relation), dim=1)
                # f2 = torch.cat((f1, cls_relation), dim=1)
                # f = torch.cat((box_feat, f2), dim=1)
            # f = torch.cat((box_feat, lung_relation), dim=1)
            # pdb.set_trace()
            # f = torch.cat((box_feat, inner_context_relation), dim=1)
            cls_score, bbox_pred = self.Box_Outs_New(f) # 新的box
            # refine_bboxes(rpn_ret['rois'], bbox_pred, im_info)
            # pdb.set_trace()

            if not self.training:
               cls_score = F.softmax(cls_score, dim=1)

        else:
            # TODO: complete the returns for RPN only situation
            pass

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss

            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))

            # pdb.set_trace()
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)

            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            # LJ
            # loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
            #     cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
            #     rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])

            # old_bbox_head loss
            # if cfg.USE_CLS_RELA or cfg.USE_SPA_RELA:
            if cfg.USE_CLS_RELA or cfg.USE_SPA_RELA or cfg.USE_CONTEXT_RELA:
                if cfg.USE_CASCADE:
                    loss_cls_old, loss_bbox_old, accuracy_cls_old = fast_rcnn_heads.fast_rcnn_losses(
                        cls_score_old, bbox_pred_old, blobs_2['labels_int32'], blobs_2['bbox_targets'],
                        blobs_2['bbox_inside_weights'], blobs_2['bbox_outside_weights'])
                else:
                    loss_cls_old, loss_bbox_old, accuracy_cls_old = fast_rcnn_heads.fast_rcnn_losses(
                        cls_score_old, bbox_pred_old, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                        rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
                # new_box_head loss
                if cfg.USE_CASCADE:
                    loss_cls_new, loss_bbox_new, accuracy_cls_new \
                        = get_loss(cls_score, bbox_pred, blobs_2)
                else:
                    loss_cls_new, loss_bbox_new, accuracy_cls_new = fast_rcnn_heads.fast_rcnn_losses(
                        cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                        rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])

                # pdb.set_trace()
                loss_cls = loss_cls_old + loss_cls_new
                loss_bbox = loss_bbox_old + loss_bbox_new
                accuracy_cls = accuracy_cls_new

            else:
                if cfg.USE_CASCADE:
                    loss_cls, loss_bbox, accuracy_cls = get_loss(cls_score, bbox_pred, blobs_2)
                else:
                    loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                        cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                        rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])

            if cfg.USE_CASCADE:
                loss_cls = loss_cls + loss_cls_cascade
                loss_bbox = loss_bbox + loss_bbox_cascade

            return_dict['losses']['loss_cls'] = loss_cls
            return_dict['losses']['loss_bbox'] = loss_bbox
            return_dict['metrics']['accuracy_cls'] = accuracy_cls

            # LJY: if the image has no mask annotations, then disable loss computing for them
            # LJ mask很小 原先错误原因 roidb 字段写错 boxes字段另有含义 现改为part_boxes
            has_mask = False
            for idx, e in enumerate(roidb):
                # pdb.set_trace()
                has_mask = e['has_mask']
                if has_mask:
                    # continue
                    break
                # ind = rpn_ret['mask_rois'][:, 0] == idx
                # rpn_ret['masks_int32'][ind, :] = np.zeros_like(rpn_ret['masks_int32'][ind, :])

            if cfg.MODEL.MASK_ON:
                if getattr(self.Mask_Head, 'SHARE_RES5', False):
                    mask_feat = self.Mask_Head(res5_feat, rpn_ret, roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                else:
                    mask_feat = self.Mask_Head(blob_conv, rpn_ret)
                    # pdb.set_trace()

                if not has_mask or np.all(rpn_ret['masks_int32'] == -1):  # LJY
                    return_dict['losses']['loss_mask'] = torch.tensor(0.0).cuda(device_id)
                else:
                    mask_pred = self.Mask_Outs(mask_feat)
                    # return_dict['mask_pred'] = mask_pred
                    # mask loss
                    loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                    return_dict['losses']['loss_mask'] = loss_mask

            if cfg.MODEL.KEYPOINTS_ON:
                if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                    # No corresponding keypoint head implemented yet (Neither in Detectron)
                    # Also, rpn need to generate the label 'roi_has_keypoints_int32'
                    kps_feat = self.Keypoint_Head(res5_feat, rpn_ret,
                                                  roi_has_keypoints_int32=rpn_ret['roi_has_keypoint_int32'])
                else:
                    kps_feat = self.Keypoint_Head(blob_conv, rpn_ret)
                kps_pred = self.Keypoint_Outs(kps_feat)
                # return_dict['keypoints_pred'] = kps_pred
                # keypoints loss
                if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'])
                else:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'],
                        rpn_ret['keypoint_loss_normalizer'])
                return_dict['losses']['loss_kps'] = loss_keypoints

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else:
            # Testing
            if not cfg.USE_CASCADE:
                return_dict['rois'] = rpn_ret['rois']
            else:
                return_dict['rois'] = new_rois_2
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, part_rois, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # FPN 模式 blobs_in 为列表 blobs_in: [fpn5, fpn4, pfn3, fpn2]
        if isinstance(blobs_in, list):
            # pdb.set_trace()
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            # LJ 保存四个尺度的下的part_roi batch=2 创建两个字典
            # 训练和测试下的batch数量不同
            if self.training:
                part_out_dic1 = {}
                part_out_dic2 = {}
            else:
                part_out_dic = {}
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                # bl_rois 与 bl_in 顺序相反 bl_rois [fpn2, fpn3, fpn4, fpn5]
                bl_rois = blob_rois + '_fpn' + str(lvl)
                # 存在某层不存在rois的情况 添加判断条件 总是提取part的feat
                if len(rpn_ret[bl_rois]) or part_rois is not None:

                    # rois 的index对应 batch的index rois包含两张图片的roi
                    # rois 不存在时为空 cat操作并不影响 此时只提取part的roi
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)

                    # LJ 添加多尺度下的 part 特征提取操作
                    if part_rois is not None:
                        # [28, 5]
                        rois_part = Variable(torch.from_numpy(part_rois)).cuda(device_id)
                        rois = torch.cat((rois, rois_part), dim=0)
                    # pdb.set_trace()
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = ROIPool((resolution, resolution), sc)(bl_in, rois)
                    elif method == 'RoIAlign':
                        xform_out = ROIAlign(
                            (resolution, resolution), sc, sampling_ratio)(bl_in, rois)

                    if part_rois is not None:
                        if cfg.USE_INNER_CONTEXT_RELA:
                            part_num = self.roi_size[0] * self.roi_size[1] * 2
                        else:
                            part_num = self.part_num
                        if self.training:
                            # len(bl_out_list) <= 4
                            if len(rpn_ret[bl_rois]):
                                bl_out_list.append(xform_out[:(-2 * part_num)])
                            # 将part的特征分解出来 [28, 256, 7, 7] 存在某个尺度下不存在rois的情况
                            part_out_dic1[blob_rois + '_fpn' + str(lvl)] = xform_out[(-2 * part_num) : (-1 * part_num)]
                            part_out_dic2[blob_rois + '_fpn' + str(lvl)] = xform_out[(-1 * part_num):]
                        else:
                            if len(rpn_ret[bl_rois]):
                                bl_out_list.append(xform_out[:(-1 * part_num)])
                            part_out_dic[blob_rois + '_fpn' + str(lvl)] = xform_out[(-1 * part_num):]
                    else:
                        bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)
            # pdb.set_trace()

            if part_rois is not None:
                # 对齐part部分的rois 取fpn4层的part feature
                if self.training:
                    xform_part = torch.cat((part_out_dic1['rois_fpn4'], part_out_dic2['rois_fpn4']), dim=0)
                else:
                    xform_part = part_out_dic['rois_fpn4']

            # Unshuffle to match rois from dataloader
            # 对齐rois 很关键 part部分的roi 也需要对齐
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
            # pdb.set_trace()

            if part_rois is not None:
                xform_out = torch.cat((xform_out, xform_part), dim=0)

        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            # pdb.set_trace()

            # 将 part_rois 与 rois 合并
            if part_rois is not None:
                rois_part = Variable(torch.from_numpy(part_rois)).cuda(device_id)
                rois = torch.cat((rois, rois_part), dim=0)
            # pdb.set_trace()
            if method == 'RoIPoolF':
                xform_out = ROIPool((resolution, resolution), spatial_scale)(blobs_in, rois)
            elif method == 'RoIAlign':
                # pdb.set_trace()
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

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
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
