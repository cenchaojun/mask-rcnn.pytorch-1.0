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
import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX
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

        # 5.31
        # categories = ['up', 'down', 'left', 'right', 'shoulder_in',
        #               'foreign_matter', 'mark_in', 'grid_shadow', 'spine_blur', 'vein_blur'
        #               ]  # 10

        self.classes = ['偏左', '偏右', '偏上', '偏下', '左肩胛骨在肺野内', '右肩胛骨在肺野内',
                        '异物', '标记在软组织或锁骨上', '栅切割伪影', '心影后脊柱不清', '双肺纹理模糊'
                        ]

        # chi2eng = {'偏上': 'up',
        #            '偏下': 'down', 'left', 'right', 'shoulder_in',
        #               'foreign_matter', 'mark_in', 'grid_shadow', 'spine_blur', 'vein_blur'
        #            }
        # self.eng_classes = ['__background__'] + [chi2eng[chi] for chi in categories]
        # self.classes = ['__background__'] + categories

        # self.classes = len(self.classes)
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'gt_classes']
        return keys

    def get_roidb(self, gt=False, crowd_filter_thresh=0):
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
        return roidb

    def _load_annotation_to_roidb(self, anno_file):
        gt_roidb = []
        entries = json.load(open(anno_file))
        '''
        filename: x, evaId: id, docName: name, fold: fold,
        classes: e.g. [0, 3, 6]
        boxes: [[x1, y1, x2, y2],
                [x1, y1, x2, y2] ...
        ]
        '''
        for entry in entries:  # for each sample in all the samples
            file_name = entry['filename'] + '.jpg'
            evaId = entry['evaId']
            image_dir = self.image_directory
            fold = entry['fold']

            cls_list = []
            for cls_name in entry['labels']:
                if cls_name not in self._class_to_ind:
                    continue
                if '肩胛骨' in cls_name:
                    cls_list.append(self._class_to_ind['左肩胛骨在肺野内'])
                    cls_list.append(self._class_to_ind['右肩胛骨在肺野内'])
                else:
                    cls_list.append(self._class_to_ind[cls_name])

            gt_boxes = entry['boxes']
            height, width = entry['height'], entry['width']

            new_entry = {'file_name': file_name, 'cls_list': cls_list, 'height': height, 'width': width,
                         'gt_boxes': gt_boxes, 'eva_id': evaId,  # 'doc_name': docName,
                         'image_dir': image_dir, 'fold': fold}
            gt_roidb.append(new_entry)

        return gt_roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(entry['image_dir'], self.image_prefix + entry['file_name'])  # LJY
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        num_valid_objs = len(entry['gt_boxes'])
        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=np.int32)

        for ix, _ in enumerate(entry['gt_boxes']):  # for each box
            boxes[ix, :] = entry['gt_boxes'][ix]
            if ix in entry['cls_list']:
                gt_classes[ix] = 1

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
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
            boxes, gt_classes = values[:2]

            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
