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
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.collections import AttrDict


def get_hospital_dataset():
    ds = AttrDict()
    # classes = [
    #     '__background__', '肺实变', '纤维化表现', '心影增大', '胸腔积液', '胸膜增厚', '主动脉异常', '膈面异常',
    #     '结节', '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '肺纹理增多', '乳头影', '弥漫性结节'
    # ]
    #
    # sents = [
    #     ['双肺野清晰，肺内未见实变影，肺门影不大，纵隔不宽，心影大小形态未见异常，两膈光滑，肋膈角锐利，'],  # 正常
    #     ['肺内未见实变影，', ''],
    #     ['', '索条影，'],
    #     ['心影大小形态未见异常，', '心影增大，'],
    #     ['两侧肋膈角锐利，', '肋膈角钝，'],
    #     ['', '胸膜增厚，'],
    #     ['', '主动脉弓迂曲，'],
    #     ['双膈面清晰，', '膈面略模糊，'],
    #     ['', '见结节状高密度影，'],
    #     ['', '见肿块影，'],
    #     ['', ''],  # 异物
    #     ['', '气胸，'],
    #     ['', '肺气肿，'],
    #     ['', '陈旧骨折，'],
    #     ['', '钙化，'],
    #     ['', '肺纹理增多，'],
    #     ['', '结节状高密度影（考虑乳头影），'],
    #     ['', '弥漫性结节高密度影，'],
    #     ['纵隔不宽，', '纵隔变宽，'],  # 纵隔
    #     ['肺门影不大，', '肺门影大，'],  # 肺门影
    #     ['', '膈面抬高，'],  # 膈面抬高
    #     ['', '脊柱侧弯，']  # 脊柱侧弯
    # ]

    # th_cls = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # threshold for each cls
    #           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    classes = [
        '__background__', '肺实变', '纤维化表现', '肋骨异常', '胸腔积液', '胸膜增厚', '主动脉异常', '膈面异常', '膈下游离气体',
        '结节', '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '乳头影', '弥漫性结节', '肺不张', '多发结节'
    ]

    chi2eng = {'肺实变': 'consolidation', '纤维化表现': 'fibrosis', '肋骨异常': 'rib_abnormity', '胸腔积液': 'effusion',
               '胸膜增厚': 'pleural_thickening', '主动脉异常': 'aorta_abnormity', '膈面异常': 'diaphragm_abnormity',
               '膈下游离气体': 'subphrenic_air', '结节': 'nodule', '肿块': 'mass', '异物': 'foreign_matters',
               '气胸': 'pneumothorax', '肺气肿': 'emphysema', '骨折': 'rib_fracture'}

    cls2chi = {cls: chi for cls, chi in enumerate(classes)}
    cls2eng = {cls: chi2eng[chi] for cls, chi in enumerate(classes)}

    eng2type = {'thoracic_asymmetry': 'thorax', 'rib_fracture': 'thorax', 'rib_destruction': 'thorax',
                'rib_abnormity': 'thorax', 'scoliosis': 'thorax',
                'diffusive_emphysema': 'lung_fields', 'circumscribed_emphysema': 'lung_fields', 'consolidation': 'lung_fields',
                'nodule': 'lung_fields', 'mass': 'lung_fields', 'fibrosis': 'lung_fields',
                'hilum_increase': 'mediastinal', 'tracheal_displacement': 'mediastinal', 'mediastinal_shift': 'mediastinal',
                'widened_mediastinal': 'mediastinal', 'cardiomegaly': 'mediastinal', 'aorta_abnormity': 'mediastinal',
                'pleural_thickening': 'pleura', 'effusion': 'pleura', 'pneumothorax': 'pleura',
                'diaphragm_abnormity': 'pleura',
                'foreign_matters': 'others', 'subphrenic_air': 'others'
                }

    # {'thorax': '胸廓', 'lung_fields': '肺部', 'mediastinal': '纵隔', 'pleura': '胸膜', 'others': '其他'}
    # cls2name = {
    #
    #     'thoracic_asymmetry': '胸廓不对称', 'rib_fracture': '肋骨骨折', 'rib_destruction': '肋骨骨质破坏',
    #     'rib_abnormity': '肋骨异常', 'scoliosis': '脊柱侧弯',
    #
    #    'diffusive_emphysema': '弥漫性肺气肿', 'circumscribed_emphysema': '局限性肺气肿', 'consolidation': '肺实变',
    #    'nodule': '结节', 'mass': '肿块', 'fibrosis': '纤维化表现',
    #
    #    'hilum_increase': '肺门影浓', 'tracheal_displacement': '气管移位', 'mediastinal_shift': '纵隔移位',
    #    'widened_mediastinal': '纵隔增宽', 'cardiomegaly': '心影增大', 'aorta_abnormity': '主动脉异常',
    #
    #    'pleural_thickening': '胸膜增厚', 'effusion': '胸腔积液', 'pneumothorax': '气胸',
    #    'diaphragm_abnormity': '膈肌异常',
    #
    #    'foreign_matters': '异物', 'subphrenic_air': '膈下游离气体'
    # }

    sents = [
        ['双肺野清晰，肺内未见实变影，肺门影不大，纵隔不宽，心影大小形态未见异常，两膈光滑，肋膈角锐利，'], # 正常
        ['肺内未见实变影，', ''],  # 肺实变
        ['', '索条影，'],  # 纤维化
        ['', '肋骨异常，'],  # 肋骨异常
        ['两侧肋膈角锐利，', '肋膈角钝，'],  # 胸腔积液
        ['', '胸膜增厚，'],  # 胸膜增厚
        ['', '主动脉弓迂曲，'],  # 主动脉异常
        ['双膈面清晰，', '膈面异常，'],  # 膈面异常
        ['', '膈下见游离气体，'],  # 膈下游离气体
        ['', '见结节状高密度影，'],  # 结节
        ['', '见肿块影，'],  # 肿块
        ['', ''],  # 异物
        ['', '气胸，'],  # 气胸
        ['', '肺气肿，'],  # 肺气肿
        ['', '骨折，'],  #  骨折
        ['', '钙化，'],  # 钙化
        ['', '结节状高密度影（考虑乳头影），'],  # 乳头影
        ['', '弥漫性结节高密度影，'],  # 弥漫性结节
        ['', '肺不张，'],  # 肺不张
        ['', '多发结节，']  # 多发结节
    ]

    th_cls = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # threshold for each cls
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.sents = sents
    ds.th_cls = th_cls
    return ds


def get_quality_dataset():
    ds = AttrDict()

    classes = ['偏左', '偏右', '偏上', '偏下', '左肩胛骨在肺野内', '右肩胛骨在肺野内',
               '异物', '标记在软组织或锁骨上', '栅切割伪影', '心影后脊柱不清', '双肺纹理模糊']

    cls2id = {name: i for i, name in enumerate(classes)}

    # chi2eng = {'肺实变': 'consolidation', '纤维化表现': 'fibrosis', '肋骨异常': 'rib_abnormity', '胸腔积液': 'effusion',
    #            '胸膜增厚': 'pleural_thickening', '主动脉异常': 'aorta_abnormity', '膈面异常': 'diaphragm_abnormity',
    #            '膈下游离气体': 'subphrenic_air', '结节': 'nodule', '肿块': 'mass', '异物': 'foreign_matters',
    #            '气胸': 'pneumothorax', '肺气肿': 'emphysema', '骨折': 'rib_fracture'}
    #
    # cls2chi = {cls: chi for cls, chi in enumerate(classes)}
    # cls2eng = {cls: chi2eng[chi] for cls, chi in enumerate(classes)}

    th_cls = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # threshold for each cls
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # ds.classes = {i: name for i, name in enumerate(classes)}
    ds.classes = classes
    ds.th_cls = th_cls
    ds.cls2id = cls2id
    return ds


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds
