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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'quality_control_train': {
        IM_DIR:
            _DATA_DIR + '/liujingyu/zhikong/fold_all',
        ANN_FN:
            _DATA_DIR + '/liujingyu/zhikong/train_anno_zhikong.json',
    },
    'PB_train': {  # dummy now, since no need to train
        IM_DIR:
            '/data1/liujingyu/PB/fold_all',
        ANN_FN:
            '/data1/liujingyu/PB/train_annos.json',
    },
    'PB_test': {
        IM_DIR:
            '/data1/liujingyu/PB/fold_all',
        ANN_FN:
            '/data1/liujingyu/PB/test_annos.json',
    },
    'hospital_train': {
        IM_DIR:
            '/data1/liujingyu/DR/fold_all',
            # '/data1/liujingyu/DR/fold_all_no_black_border', # '/lung_plain_data/liujingyu/DX/fold_all',
        ANN_FN:
            # '/home/lianjie/deepwise_x-ray/jsons/train_x-ray.json',
            # '/home/lianjie/cvpr_code/part_seg/jsons/train_annos_part_20200114.json',
            # '/data1/liujingyu/DR/train_annos_PB+neg.json',
            '/home/lianjie/cvpr_code/part_seg/found_yy_jsons/train_gk_yy.json', # '/data1/DX/train_anno_DX.json',
    },
    'hospital_test': {
        IM_DIR:
            # '/data1/liujingyu/DR/fold_all_no_black_border', # '/lung_plain_data/liujingyu/DX/fold_all',
            '/data1/liujingyu/DR/fold_all',
        ANN_FN:
            # '/home/lianjie/deepwise_x-ray/jsons/test_x-ray.json',
            # '/home/lianjie/cvpr_code/part_seg/jsons/test_annos_part_20200114.json',
            # '/data1/liujingyu/DR/test_annos_PB+neg.json',
            '/home/lianjie/cvpr_code/part_seg/found_yy_jsons/test_gk_yy.json', # '/data1/DX/test_anno_DX.json',
    },
    'hospital_train_part': {
        IM_DIR:
            _DATA_DIR + '/DX/fold1-7_part',
        ANN_FN:
            _DATA_DIR + '/DX/train_anno_DX_part.json',
    },
    'hospital_train_small': {
        IM_DIR:
            _DATA_DIR + '/DX/fold_all_small',
        ANN_FN:
            _DATA_DIR + '/DX/train_anno_DX_small.all.json',
    },
    'hospital_test_part': {
        IM_DIR:
            _DATA_DIR + '/DX/fold1-7_part',
        ANN_FN:
            _DATA_DIR + '/DX/test_anno_DX_part.json',
    },
    'hospital_test_small': {
        IM_DIR:
            _DATA_DIR + '/DX/fold_all_small',
        ANN_FN:
            _DATA_DIR + '/DX/test_anno_DX_small.all.json',
    },
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_train': {
        IM_DIR:
            # _DATA_DIR + '/coco2017/train2017',
            'data/coco2017/train2017',
        ANN_FN:
            # _DATA_DIR + '/coco2017/annotations_trainval2017/instances_train2017.json',
            'data/coco2017/annotations_trainval2017/instances_train2017.json',
    },
    'coco_2017_val': {
        IM_DIR:
            # _DATA_DIR + '/coco/images/val2017',
            'data/coco2017/val2017',
        ANN_FN:
            # _DATA_DIR + '/coco/annotations/instances_val2017.json',
            'data/coco2017/annotations_trainval2017/instances_val2017.json',
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json'
    },
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'keypoints_coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    }
}
