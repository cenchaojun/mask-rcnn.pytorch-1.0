# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import json
import pdb

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX
from itertools import chain
from copy import deepcopy
import pdb
logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        if type(DATASETS[name][ANN_FN]) == list:
            for _ in DATASETS[name][ANN_FN]:
                assert os.path.exists(_), 'Annotation file \'{}\' not found'.format(_)
        else:
            assert os.path.exists(DATASETS[name][ANN_FN]), 'Annotation file \'{}\' not found'.\
                format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.debug_timer = Timer()

        # 10.31
        # categories = [
        #     '肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '主动脉结增宽', '膈面异常', '结节',  # 1 ~ 7
        #     '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '乳头影', '弥漫性结节', '肺不张', '多发结节',  # 8 ~ 17
        #     '心影增大', '脊柱侧弯', '纵隔变宽', '肺门增浓', '膈下游离气体', '肋骨异常',  # 18 ~ 23
        #     '皮下气肿', '主动脉钙化', '空洞', '液气胸', '肋骨缺失', '肩关节异常', '肺纹理增多'  # 24 ~ 30
        # ]  # 30

        # cvpr code
        categories = ['肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '结节',
                      '肿块', '气胸', '肺气肿', '钙化', '弥漫性结节', '肺不张',
                      '心影增大', '骨折']  #
        # 竞赛
        # categories = ['肺实变', '纤维化表现', '胸腔积液', '结节',
        #               '肿块', '气胸', '肺气肿', '钙化', '肺不张', '骨折']

        # 肺结核实验
        # categories = ['肺结核']

        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        # LJ add parts info
        # cvpr code
        self.parts = ['左肩胛骨', '右肩胛骨', '左肺', '右肺', '心影']
        self._class_to_parts = dict(zip(self.parts, range(len(self.parts))))

    @property
    def cache_path(self):
        #cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        cache_path = '/home/lianjie/mask-rcnn.pytorch-1.0/jeline_output/cache'
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'segms', 'gt_classes', 'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map']
        return keys

    def get_roidb(
            self,
            gt=False,
            crowd_filter_thresh=0
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'

        roidb = self._load_annotation_to_roidb(DATASETS[self.name][ANN_FN])
        for entry in roidb:
            self._prep_roidb_entry(entry)
        # pdb.set_trace()
        if gt:
            # Include ground-truth object annotations
            cache_filepath = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
            if os.path.exists(cache_filepath) and not cfg.DEBUG:
                self.debug_timer.tic()
                self._add_gt_from_cache(roidb, cache_filepath)
                logger.debug(
                    '_add_gt_from_cache took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
            else:
                self.debug_timer.tic()
                for entry in roidb:
                    self._add_gt_annotations(entry)
                logger.debug(
                    '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
                # save roidb file
                with open(cache_filepath, 'wb') as fp:
                    pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
                logger.info('Cache ground truth roidb to %s', cache_filepath)
        _add_class_assignments(roidb)
        # pdb.set_trace()
        return roidb

    def _valid_polygon(self, polygon):
        if len(polygon) < 3:
            return False
        for point in polygon:
            for xy in point:
                if xy < 0:
                    return False
        return True

    def _load_annotation_to_roidb(self, anno_file):
        gt_roidb = []
        entries = json.load(open(anno_file))
        # for imageId, entry in entries.items():  # for each sample in all the samples
        small_cnt = 0
        for entry in entries:  # for each sample in all the samples
            # pdb.set_trace()
            syms = entry['syms']
            polygons = entry['polygon']

            if 'boxes' in entry:
                boxes = entry['boxes']
            else:
                boxes = []

            if 'rows' in entry:
                rows = entry['rows']
                cols = entry['cols']
            elif 'bottom_row' in entry:
                 rows = entry['bottom_row'] - entry['top_row']
                 cols = entry['right_col'] - entry['left_col']

            # 新版json 关键字变化 file_name eva_id doc_name
            file_name = entry['file_name']
            # if file_name in ['58695.png', '57569.png', '45795.png', '59788.png', '60191.png',
            #         '60795.png', '69838.png', '70454.png', '69845.png']:
            #     continue

            image_dir = self.image_directory
            evaId = entry['eva_id']
            docName = entry['doc_name']
            fold = entry['fold']

            if 'offset_x' in entry and 'offset_y' in entry:
                offset_x, offset_y = entry['offset_x'], entry['offset_y']
            else:
                offset_x, offset_y = 0, 0

            has_mask = False
            if len(polygons) > 0 and len(boxes) == 0:
                has_mask = True
            elif len(boxes) > 0 and len(polygons) == 0:
                has_mask = False
                for box in boxes:
                    x1, y1, x2, y2 = box
                    polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    polygons.append(polygon)

            npoly = len(polygons)
            is_valid_polygon = [True for _ in range(npoly)]

            # count number of valid syms
            for idx, (sym, polygon) in enumerate(zip(syms, polygons)):
                if not self._valid_polygon(polygon):  # exclude non-valid polygons
                    is_valid_polygon[idx] = False
                    continue

            cls_list, gt_boxes = [], []
            filtered_polygons = []

            remove_flag = False
            for idx, (sym, polygon) in enumerate(zip(syms, polygons)):  # for each sym or region
                if not is_valid_polygon[idx]:
                    continue
                if '膈面异常' in sym and entry['doc_name'] == 'fj6311':
                    continue
                # if '胸腔积液' in sym and entry['file_name'] == '39420.png':
                #     continue

                if '主动脉异常' in sym and '钙化' in sym:
                    sym = ['主动脉钙化', '主动脉异常']
                if '结节' in sym and '乳头影' in sym:  # 费主任标了好多这种，结节和乳头影都在，我们认为是乳头影
                    sym = ['乳头影']

                if '结节' in sym and '弥漫性结节' in sym:
                    sym.remove('结节')
                if '结节' in sym and '多发结节' in sym:
                    sym.remove('结节')
                if '结核结节' in sym and '弥漫性结节' in sym:
                    sym.remove('结核结节')
                if '结核结节' in sym and '多发结节' in sym:
                    sym.remove('结核结节')
                if '结核球' in sym and '弥漫性结节' in sym:
                    sym.remove('结核球')
                if '结核球' in sym and '多发结节' in sym:
                    sym.remove('结核球')

                for s in sym:  # for each sub-sym
                    if s == '膈面膨隆' or s == '膈面抬高':  # awkward ...
                        s = '膈面异常'
                    # if s == '盘状肺不张':
                    #     s = '纤维化表现'
                    # if s == '肺结核':  # ignore 肺结核
                    #     s = '肺实变'
                    if s == '肺门影浓' or s == '肺门影大':
                        s = '肺门增浓'
                    if s == '主动脉异常':
                        s = '主动脉结增宽'

                    # 以下是肺结核的征象
                    if s == '三均匀粟粒样结节' or s == '非三均匀粟粒样结节':
                        s = '弥漫性结节'
                    if s == '结核球' or s == '结核结节':
                        s = '结节'
                    if s == '索条影':
                        s = '纤维化表现'

                    # cvpr code
                    if s == '骨折' or s == '肋骨缺失':
                        s = '骨折'
                    if s == '弥漫性结节' or s == '多发结节':
                        s = '弥漫性结节'

                    if s in self._class_to_ind:  # if in the given classes
                        polygon = [tuple(point) for point in polygon]
                        polygon_np = np.array(polygon)
                        x1, y1, x2, y2 = polygon_np[:, 0].min(), polygon_np[:, 1].min(), \
                                         polygon_np[:, 0].max(), polygon_np[:, 1].max(),
                        x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                            x1, y1, x2, y2, rows, cols
                        )

                        # # if nodule is too large, then assign it to diffusive nodules
                        if s == '结节' and (x2 - x1 > 300 or y2 - y1 > 300):
                            s = '弥漫性结节'
                            # remove_flag = True
                            # break

                        cls = self._class_to_ind[s]
                        cls_list.append(cls)

                        # expand too-small boxes (width or height < 20)
                        if x2 - x1 < 20:
                            cx = (x1 + x2) * 0.5
                            x1 = cx - 10
                            x2 = cx + 10
                            small_cnt += 1
                        if y2 - y1 < 20:
                            cy = (y1 + y2) * 0.5
                            y1 = cy - 10
                            y2 = cy + 10
                            small_cnt += 1

                        gt_boxes.append([x1, y1, x2, y2])
                        tmp = [list(chain.from_iterable(polygon))]  # [[x1, y1], [x2, y2]] -> [[x1, y1, x2, y2]]
                        filtered_polygons.append(tmp)
                    else:
                        # print(s)
                        pass

            if not remove_flag:
                assert len(cls_list) == len(gt_boxes) == len(filtered_polygons)
                # LJ add parts info
                parts_list = []
                parts = entry['parts']
                part_boxes = entry['part_boxes']
                for part_name in parts:
                    parts_list.append(self._class_to_parts[part_name])
                part_boxes = np.array(part_boxes, dtype=np.float32)
                parts_list = np.array(parts_list, dtype=np.int32)

                new_entry = {'file_name': file_name, 'cls_list': cls_list, 'height': rows, 'width': cols,
                             'polygons': filtered_polygons, 'gt_boxes': gt_boxes, 'eva_id': evaId, 'doc_name': docName,
                             'image_dir': image_dir, 'offset_x': offset_x, 'offset_y': offset_y, 'fold': fold,
                             'has_mask': has_mask, 'parts_list': parts_list, 'part_boxes': part_boxes}
                gt_roidb.append(new_entry)

        return gt_roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        # im_path = os.path.join(self.image_directory, self.image_prefix + entry['file_name'])
        im_path = os.path.join(entry['image_dir'], self.image_prefix + entry['file_name'])  # LJY
        # pdb.set_trace()
        # print(entry['fold'])
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        # LJ add parts info
        # entry['part_classes'] = np.empty((0), dtype=np.int32)
        # entry['part_boxes'] = np.empty((0, 4), dtype=np.float32)

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        num_valid_objs = len(entry['polygons'])

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=np.int32)
        gt_overlaps = np.zeros((num_valid_objs, self.num_classes), dtype=entry['gt_overlaps'].dtype)
        # seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros((num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype)

        for ix, _ in enumerate(entry['polygons']):
            # pdb.set_trace()
            boxes[ix, :] = entry['gt_boxes'][ix]
            cls = entry['cls_list'][ix]
            gt_classes[ix] = cls
            # seg_areas[ix] = obj['area']
            box_to_gt_ind_map[ix] = ix
            gt_overlaps[ix, cls] = 1.0

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(entry['polygons'])
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        # entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(entry['gt_overlaps'].toarray(), gt_overlaps, axis=0)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(entry['box_to_gt_ind_map'], box_to_gt_ind_map)
        del entry['cls_list']
        del entry['gt_boxes']

    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        print(len(roidb), len(cached_roidb))
        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            boxes, segms, gt_classes, gt_overlaps, is_crowd, box_to_gt_ind_map = values[:6]

            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['segms'].extend(segms)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            # entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )
            # LJ
            # entry['has_mask'] = cached_entry['has_mask']


def snip_valid(im_size, boxes):
    # short_side = im_size.min()
    # if short_side == 600:  # small boxes are invalid in the minimum scale
    #     ind = np.where((boxes[:, 2] - boxes[:, 0] > 50) and (boxes[:, 3] - boxes[:, 1] > 50))
    # elif short_side == 1600:  # large boxes are invalid in the maximum scale
    #     ind = np.where((boxes[:, 2] - boxes[:, 0] < 800) and (boxes[:, 2] - boxes[:, 0] < 800))
    # else:  # we assume they are valid always valid
    #     pass

    ind = (0 < boxes[:, 2] - boxes[:, 0]) & (boxes[:, 2] - boxes[:, 0] < 3000) & \
          (0 < boxes[:, 3] - boxes[:, 1]) & (boxes[:, 3] - boxes[:, 1] < 3000)

    return ind


def add_proposals(roidb, rois, scales, im_sizes):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)  # scale the rois back
    _merge_proposal_boxes_into_roidb(roidb, box_list, scales, im_sizes)  # LJY: call the function below

    _add_class_assignments(roidb)  # LJY: call the function below


def _merge_proposal_boxes_into_roidb(roidb, box_list, scales=None, im_sizes=None):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)

    # drop some invalid boxes based on SNIP
    for i, entry in enumerate(roidb):  # for each image
        scaled_gt_boxes = entry['boxes'] * scales[i]
        ind = snip_valid(im_sizes[i], scaled_gt_boxes)

        # if len(ind) - np.count_nonzero(ind) > 0:  # indeed drop
        #     pdb.set_trace()

        num_remain = np.count_nonzero(ind)
        box_to_gt_ind_map = np.zeros(num_remain, dtype=entry['box_to_gt_ind_map'].dtype)
        for i in range(num_remain):
            box_to_gt_ind_map[i] = i
        entry['box_to_gt_ind_map'] = box_to_gt_ind_map

        entry['boxes'], entry['gt_classes'], entry['gt_overlaps'], entry['is_crowd'] = \
            entry['boxes'][ind], entry['gt_classes'][ind], entry['gt_overlaps'][ind], entry['is_crowd'][ind]

        ind_list = np.where(ind is True)[0].tolist()
        entry['segms'] = [entry['segms'][k] for k in ind_list]

    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]

        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]
