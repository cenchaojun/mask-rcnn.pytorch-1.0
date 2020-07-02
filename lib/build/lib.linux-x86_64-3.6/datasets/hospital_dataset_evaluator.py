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

"""Functions for evaluating results computed for a json dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import numpy as np
import os
import os.path as osp
import uuid
import pdb

from pycocotools.cocoeval import COCOeval

from core.config import cfg
from utils.io import save_object
import utils.boxes as box_utils
from utils.other_utils import bbox_overlaps, compute_AUCs
from utils.vis import vis_boxes_ljy

import cv2
import torch

logger = logging.getLogger(__name__)


def evaluate_masks(
    json_dataset,
    all_boxes,
    all_segms,
    output_dir,
    use_salt=True,
    cleanup=False
):
    res_file = os.path.join(
        output_dir, 'segmentations_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_segms_results_file(
        json_dataset, all_boxes, all_segms, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if json_dataset.name.find('test') == -1:
        coco_eval = _do_segmentation_eval(json_dataset, res_file, output_dir)
    else:
        coco_eval = None
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


# mapping from class name to threshold, should be in pace with product usage
cls2th = {
    '肺实变': 0.5,
    '纤维化表现': 0.5,
    '胸腔积液': 0.6,
    '胸膜增厚': 0.5,
    '主动脉结增宽': 0.8,
    '膈面异常': 0.8,
    '结节': 0.5,
    '肿块': 0.6,
    '异物': 0.7,
    '气胸': 0.5,
    '肺气肿': 0.5,
    '骨折': 0.5,
    '钙化': 0.5,
    '乳头影': 0.5,
    '弥漫性结节': 0.6,
    '肺不张': 0.5,
    '多发结节': 0.6,
    '心影增大': 0.8,
    '脊柱侧弯': 0.5,
    '纵隔变宽': 0.5,
    '肺门增浓': 0.7,
    '膈下游离气体': 0.7,
    '肋骨异常': 0.5,
    '肺结核': 0.6,
    '皮下气肿': 0.7,
    '主动脉钙化': 0.6
}


def _write_coco_segms_results_file(json_dataset, all_boxes, all_segms, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "segmentation": [...],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_segms_results_one_category(
            json_dataset, all_boxes[cls_ind], all_segms[cls_ind], cat_id))
    logger.info(
        'Writing segmentation results json to: {}'.format(
            os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _coco_segms_results_one_category(json_dataset, boxes, segms, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(boxes) == len(image_ids)
    assert len(segms) == len(image_ids)
    for i, image_id in enumerate(image_ids):
        dets = boxes[i]
        rles = segms[i]

        if isinstance(dets, list) and len(dets) == 0:
            continue

        dets = dets.astype(np.float)
        scores = dets[:, -1]

        results.extend(
            [{'image_id': image_id,
              'category_id': cat_id,
              'segmentation': rles[k],
              'score': scores[k]}
              for k in range(dets.shape[0])])

    return results


def _do_segmentation_eval(json_dataset, res_file, output_dir):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'segmentation_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    return coco_eval


def evaluate_boxes(json_dataset, all_boxes, output_dir, use_salt=True, cleanup=False):
    res_file = os.path.join(output_dir, 'bbox_' + json_dataset.name + '_results')
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_bbox_results_file(json_dataset, all_boxes, res_file)

    # pdb.set_trace()
    coco_eval = _do_detection_eval(json_dataset, res_file, output_dir)

    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


def evaluate_mAP_combine(json_datasets, roidbs, all_boxes_list, output_dir, cleanup=False):
    """ LJY
    all_boxes: num_cls x num_images x [num_boxes x 5]
    """
    json_dataset = json_datasets[0]
    mAP_folder = '/home/liujingyu/code/mAP'

    roidb, roidb_part = roidbs[0], roidbs[1]
    all_boxes, all_boxes_part = all_boxes_list[0], all_boxes_list[1]

    small_classes = ['结节', '肺实变', '膈面异常', '骨折']

    for i, (entry, entry_part) in enumerate(zip(roidb, roidb_part)):  # for each pair of images
        # print(i, entry['eva_id'], entry['file_name'])
        assert entry['file_name'] == entry_part['file_name']

        file_name = entry['file_name'][:-4] + '.txt'
        fgt = open(osp.join(mAP_folder, 'ground-truth', file_name), 'w')
        fpred = open(osp.join(mAP_folder, 'predicted', file_name), 'w')

        for cls_ind, cls in enumerate(json_dataset.classes):  # for each cls
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break

            gt_classes = roidb[i]['gt_classes']
            ind = np.where(gt_classes == cls_ind)

            gt_boxes = roidb[i]['boxes'][ind]

            dets = all_boxes[cls_ind][i]  # N x 5, [x1, y1, x2, y2, score]
            dets_part = all_boxes_part[cls_ind][i]  # N x 5

            # offset the dets_part based on offset
            dets_part[:, 0] += entry_part['offset_x']
            dets_part[:, 2] += entry_part['offset_x']
            dets_part[:, 1] += entry_part['offset_y']
            dets_part[:, 3] += entry_part['offset_y']

            # select
            # if cls in small_classes:
            #     dets = dets_part

            # merge
            dets = np.vstack((dets, dets_part))

            # NMS on dets and dets_part
            if cfg.TEST.SOFT_NMS.ENABLED:
                nms_dets, _ = box_utils.soft_nms(
                    dets,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=cfg.TEST.NMS,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:
                keep = box_utils.nms(dets, cfg.TEST.NMS)
                nms_dets = dets[keep, :]
            # Refine the post-NMS boxes using bounding-box voting
            if cfg.TEST.BBOX_VOTE.ENABLED:
                nms_dets = box_utils.box_voting(
                    nms_dets,
                    dets,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )
            dets = nms_dets

            # write gt boxes, format: tvmonitor 2 10 173 238
            for k in range(gt_boxes.shape[0]):
                s = '{} {:f} {:f} {:f} {:f}'.format(cls, gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2], gt_boxes[k, 3])
                fgt.write(s)
                fgt.write('\n')

                if cls == '肿块' or cls == '结节' or cls == '钙化' or cls == '乳头影':
                    s = '{} {:f} {:f} {:f} {:f}'.format('肿块结节钙化',
                                                        gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2], gt_boxes[k, 3])
                    fgt.write(s)
                    fgt.write('\n')

                if cls == '纤维化表现' or cls == '肺实变' or cls == '肺纹理增多' or cls == '肿块' or cls == '弥漫性结节':
                    s = '{} {:f} {:f} {:f} {:f}'.format('高密度影',
                                                        gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2], gt_boxes[k, 3])
                    fgt.write(s)
                    fgt.write('\n')

                if cls == '气胸' or cls == '气肿':
                    s = '{} {:f} {:f} {:f} {:f}'.format('低密度影',
                                                        gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2], gt_boxes[k, 3])
                    fgt.write(s)
                    fgt.write('\n')

            # write pred boxes, format: tvmonitor 0.471781 0 13 174 244
            for k in range(dets.shape[0]):
                s = '{} {:f} {:f} {:f} {:f} {:f}'.format(cls, dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3])
                fpred.write(s)
                fpred.write('\n')

                if cls == '肿块' or cls == '结节' or cls == '钙化' or cls == '乳头影':
                    s = '{} {:f} {:f} {:f} {:f} {:f}'.format('肿块结节钙化',
                                                             dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3])
                    fpred.write(s)
                    fpred.write('\n')

                if cls == '纤维化表现' or cls == '肺实变' or cls == '肺纹理增多' or cls == '肿块' or cls == '弥漫性结节':
                    s = '{} {:f} {:f} {:f} {:f} {:f}'.format('高密度影',
                                                             dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3])
                    fpred.write(s)
                    fpred.write('\n')

                if cls == '气胸' or cls == '气肿':
                    s = '{} {:f} {:f} {:f} {:f} {:f}'.format('低密度影',
                                                             dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3])
                    fpred.write(s)
                    fpred.write('\n')
            if gt_boxes.shape[0] > 0:  # then we draw gt boxes and pred boxes
                im = cv2.imread(entry['image'])
                more_text = str(entry['eva_id']) + ' ' + entry['doc_name']
                im = vis_boxes_ljy(im, gt_boxes, dets[:, :-1], more_text)
                out_path = os.path.join('/data5/liujingyu/mask_rcnn_Outputs/vis', cls, entry['file_name'])
                cv2.imwrite(out_path, im)

    pdb.set_trace()


def evaluate_mAP_manufacturer(json_datasets, roidbs, all_boxes_list, output_dir, cleanup=False):
    """ LJY
    all_boxes: num_cls x num_images x [num_boxes x 5]
    """
    mAP_folder = '/home/liujingyu/code/mAP'
    json_dataset = json_datasets[0]
    all_boxes = all_boxes_list[0]
    roidb = roidbs[0]

    for i, entry in enumerate(roidb):  # for each image
        print(i, entry['eva_id'], entry['file_name'])
        file_name = entry['file_name'][:-4] + '.txt'
        manufacturer, manufacturer_model = entry['manufacturer'], entry['manufacturer_model']

        if manufacturer is not None:
            # pdb.set_trace()
            manufacturer, manufacturer_model = entry['manufacturer'].strip('\"'), entry['manufacturer_model'].strip('\"')
            if not osp.exists(osp.join(mAP_folder, 'ground-truth_' + manufacturer)):
                os.makedirs(osp.join(mAP_folder, 'ground-truth_' + manufacturer))
                os.makedirs(osp.join(mAP_folder, 'predicted_' + manufacturer))

            fgt = open(osp.join(mAP_folder, 'ground-truth_' + manufacturer, file_name), 'w')
            fpred = open(osp.join(mAP_folder, 'predicted_' + manufacturer, file_name), 'w')

            for cls_ind, cls in enumerate(json_dataset.classes):  # for each cls
                if cls == '__background__':
                    continue
                if cls_ind >= len(all_boxes):
                    break

                gt_classes = roidb[i]['gt_classes']
                ind = np.where(gt_classes == cls_ind)

                gt_boxes = roidb[i]['boxes'][ind]
                dets = all_boxes[cls_ind][i]

                # write gt files, format: tvmonitor 2 10 173 238
                for k in range(gt_boxes.shape[0]):
                    s = '{} {:f} {:f} {:f} {:f}'.format(cls, gt_boxes[k, 0], gt_boxes[k, 1],
                                                             gt_boxes[k, 2], gt_boxes[k, 3])
                    fgt.write(s)
                    fgt.write('\n')

                # write pred files, format: tvmonitor 0.471781 0 13 174 244
                for k in range(dets.shape[0]):
                    s = '{} {:f} {:f} {:f} {:f} {:f}'.format(cls, dets[k, -1], dets[k, 0], dets[k, 1],
                                                                  dets[k, 2], dets[k, 3])
                    fpred.write(s)
                    fpred.write('\n')

                if gt_boxes.shape[0] > 0:  # then we draw gt box and pred boxes
                    im = cv2.imread(entry['image'])
                    more_text = str(entry['eva_id']) + ' ' + entry['doc_name']
                    im = vis_boxes_ljy(im, gt_boxes, dets[:, :-1], more_text)
                    out_path = os.path.join('Outputs/vis', cls, entry['file_name'])
                    cv2.imwrite(out_path, im)

    pdb.set_trace()


def evaluate_mAP(json_datasets, roidbs, all_boxes_list, output_dir, cleanup=False):
    """ LJY
    all_boxes: num_cls x num_images x [num_boxes x 5]
    """
    mAP_folder = '/home/liujingyu/code/mAP'
    # mAP_folder = '/lung_plain_data/liujingyu/code/mAP'
    json_dataset = json_datasets[0]
    # pdb.set_trace()
    all_boxes = all_boxes_list[0]
    roidb = roidbs[0]

    for i, entry in enumerate(roidb):  # for each image
        print(i, entry['eva_id'], entry['fold'], entry['file_name'])
        file_name = entry['file_name'][:-4] + '.txt'

        fgt = open(osp.join(mAP_folder, 'ground-truth', file_name), 'w')
        fpred = open(osp.join(mAP_folder, 'predicted', file_name), 'w')

        # pdb.set_trace()
        fold, doc_name = entry['fold'], entry['doc_name']
        if 'gongkai' not in fold:
            fgt_good = open(osp.join(mAP_folder, 'ground-truth-good', file_name), 'w')
            fpred_good = open(osp.join(mAP_folder, 'predicted-good', file_name), 'w')
        if doc_name == 'yangyan123':
            fgt_yang = open(osp.join(mAP_folder, 'ground-truth-yang', file_name), 'w')
            fpred_yang = open(osp.join(mAP_folder, 'predicted-yang', file_name), 'w')

        for cls_ind, cls in enumerate(json_dataset.classes):  # for each cls
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break

            gt_classes = entry['gt_classes']
            ind = np.where(gt_classes == cls_ind)

            gt_boxes = entry['boxes'][ind]
            dets = all_boxes[cls_ind][i]

            # write gt files, format: tvmonitor 2 10 173 238
            for k in range(gt_boxes.shape[0]):
                s = '{} {:f} {:f} {:f} {:f}'.format(cls, gt_boxes[k, 0], gt_boxes[k, 1],
                                                         gt_boxes[k, 2], gt_boxes[k, 3])
                fgt.write(s)
                fgt.write('\n')
                if 'gongkai' not in fold:
                    fgt_good.write(s)
                    fgt_good.write('\n')
                if doc_name == 'yangyan123':
                    fgt_yang.write(s)
                    fgt_yang.write('\n')

                # s = '{} {:f} {:f} {:f} {:f}'.format('所有征象',
                #                                     gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2],
                #                                     gt_boxes[k, 3])
                # fgt.write(s)
                # fgt.write('\n')
                # if cls == '结节' or cls == '钙化' or cls == '乳头影':
                #     s = '{} {:f} {:f} {:f} {:f}'.format('结节钙化乳头影',
                #                                         gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2],
                #                                         gt_boxes[k, 3])
                #     fgt.write(s)
                #     fgt.write('\n')
                # if cls == '纤维化表现' or cls == '肺实变' or cls == '肺纹理增多' or cls == '肿块' or cls == '肺不张' or \
                #    cls == '胸腔积液' or cls == '胸膜增厚':
                #     s = '{} {:f} {:f} {:f} {:f}'.format('高密度影',
                #                                         gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2],
                #                                         gt_boxes[k, 3])
                #     fgt.write(s)
                #     fgt.write('\n')
                # if cls == '弥漫性结节' or cls == '多发结节':
                #     s = '{} {:f} {:f} {:f} {:f}'.format('弥漫多发结节',
                #                                         gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2],
                #                                         gt_boxes[k, 3])
                #     fgt.write(s)
                #     fgt.write('\n')
                # if cls == '骨折' or cls == '肋骨异常':
                #     s = '{} {:f} {:f} {:f} {:f}'.format('肋骨骨折异常',
                #                                         gt_boxes[k, 0], gt_boxes[k, 1], gt_boxes[k, 2],
                #                                         gt_boxes[k, 3])
                #     fgt.write(s)
                #     fgt.write('\n')

            # write pred files, format: tvmonitor 0.471781 0 13 174 244
            for k in range(dets.shape[0]):
                s = '{} {:f} {:f} {:f} {:f} {:f}'.format(cls, dets[k, -1], dets[k, 0], dets[k, 1],
                                                              dets[k, 2], dets[k, 3])
                fpred.write(s)
                fpred.write('\n')
                if 'gongkai' not in fold:
                    fpred_good.write(s)
                    fpred_good.write('\n')
                if doc_name == 'yangyan123':
                    fpred_yang.write(s)
                    fpred_yang.write('\n')

                # s = '{} {:f} {:f} {:f} {:f} {:f}'.format('所有征象',
                #                                          dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3])
                # fpred.write(s)
                # fpred.write('\n')
                # if cls == '结节' or cls == '钙化' or cls == '乳头影':
                #     s = '{} {:f} {:f} {:f} {:f} {:f}'.format('结节钙化乳头影',
                #                                              dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3])
                #     fpred.write(s)
                #     fpred.write('\n')
                #
                # if cls == '纤维化表现' or cls == '肺实变' or cls == '肺纹理增多' or cls == '肿块' or cls == '肺不张' or \
                #    cls == '胸腔积液' or cls == '胸膜增厚':
                #     s = '{} {:f} {:f} {:f} {:f} {:f}'.format('高密度影',
                #                                              dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3])
                #     fpred.write(s)
                #     fpred.write('\n')
                #
                # if cls == '弥漫性结节' or cls == '多发结节':
                #     s = '{} {:f} {:f} {:f} {:f} {:f}'.format('弥漫多发结节',
                #                                              dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2],
                #                                              dets[k, 3])
                #     fpred.write(s)
                #     fpred.write('\n')
                #
                # if cls == '骨折' or cls == '肋骨异常':
                #     s = '{} {:f} {:f} {:f} {:f} {:f}'.format('肋骨骨折异常',
                #                                              dets[k, -1], dets[k, 0], dets[k, 1], dets[k, 2],
                #                                              dets[k, 3])
                #     fpred.write(s)
                #     fpred.write('\n')

            # if gt_boxes.shape[0] > 0:  # then we draw gt box and pred boxes
            #     im = cv2.imread(entry['image'])
            #     more_text = str(entry['eva_id']) + ' ' + entry['doc_name']
            #     im = vis_boxes_ljy(im, gt_boxes, dets[:, :-1], more_text)
            #     if not os.path.exists('Outputs/vis/' + cls):
            #         os.makedirs('Outputs/vis/' + cls)
            #
            #     out_path = os.path.join('Outputs/vis', cls, entry['file_name'])
            #     cv2.imwrite(out_path, im)
            # elif dets.shape[0] > 0:  # draw fp
            #     im = cv2.imread(entry['image'])
            #     more_text = str(entry['eva_id']) + ' ' + entry['doc_name']
            #     im = vis_boxes_ljy(im, gt_boxes, dets[:, :-1], more_text)
            #
            #     if not os.path.exists('Outputs/vis/' + cls + '_fp'):
            #         os.makedirs('Outputs/vis/' + cls + '_fp')
            #
            #     out_path = os.path.join('Outputs/vis', cls + '_fp', entry['file_name'])
            #     cv2.imwrite(out_path, im)

    pdb.set_trace()


def evaluate_boxes_recall_and_FP(json_dataset, roidb, all_boxes, output_dir, cleanup=False):
    """ LJY
    all_boxes: num_cls x num_images x [num_boxes x 5]
    """
    res_file = os.path.join(output_dir, 'bbox_' + json_dataset.name + '_results')
    res_file += '.json'

    GTs = [0 for _ in json_dataset.classes]
    TPs = [0 for _ in json_dataset.classes]
    FPs = [0 for _ in json_dataset.classes]
    NPreds = [0 for _ in json_dataset.classes]
    FP_imgs = [0 for _ in json_dataset.classes]  # num of images that have FPs

    for i, entry in enumerate(roidb):  # for each image
        for cls_ind, cls in enumerate(json_dataset.classes):  # for each cls
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break

            gt_classes = roidb[i]['gt_classes']
            ind = np.where(gt_classes == cls_ind)

            gt_boxes = roidb[i]['boxes'][ind]
            dets = all_boxes[cls_ind][i]
            scores = dets[:, -1]
            pred_boxes = dets[scores > 0.3, :4]  # xyxy

            IOUs = bbox_overlaps(pred_boxes, gt_boxes)  # N x 4, K x 4 -> N x K
            npred, ngt = pred_boxes.shape[0], gt_boxes.shape[0]

            hit_map = IOUs > 0.01
            tp = hit_map.any(axis=0).sum()  # num of true positive of this cls, this image
            fp = npred - hit_map.any(axis=1).sum()  # num of false positive of this cls, this image

            GTs[cls_ind] += ngt
            TPs[cls_ind] += tp
            FPs[cls_ind] += fp
            NPreds[cls_ind] += npred

            if fp > 0:
                FP_imgs[cls_ind] += 1

            # pdb.set_trace()
            if ngt > 0:  # then we draw gt box and pred boxes
                im = cv2.imread(entry['image'])
                more_text = str(entry['eva_id']) + ' ' + entry['doc_name']
                im = vis_boxes_ljy(im, gt_boxes, pred_boxes, more_text)
                out_path = os.path.join('/data5/liujingyu/mask_rcnn_Outputs/vis', cls, entry['file_name'])
                cv2.imwrite(out_path, im)

    print('# GT on instance level:')
    for GT in GTs[1:]:
        print(GT)

    print('recall on instance level')
    for TP, GT in zip(TPs[1:], GTs[1:]):
        print("%.3f" % (TP / GT))

    print('FPs per image')
    for FP, NPred in zip(FPs[1:], NPreds[1:]):
        print("%.3f" % (FP / len(roidb)))

    pdb.set_trace()

    # print('# images have FP:')
    # for FP in FP_imgs[1:]:
    #     print(FP)


def evaluate_class_AUC(json_dataset, roidb, all_boxes, output_dir, cleanup=False):
    """ LJY """
    # pdb.set_trace()
    res_file = os.path.join(output_dir, 'bbox_' + json_dataset.name + '_results')
    res_file += '.json'

    num_classes = len(json_dataset.classes)

    gt_all = np.zeros((len(roidb), num_classes), dtype=np.uint8)
    pred_all = np.zeros((len(roidb), num_classes), dtype=np.float32)
    scores_cls = [[] for _ in range(num_classes)]
    Th_cls = [0 for _ in range(num_classes)]

    hit = [0 for _ in range(num_classes)]
    FP = [0 for _ in range(num_classes)]

    # We fix the recall of each class to 0.9, then compute FP. First thing is to get the threshold for each cls
    for i, entry in enumerate(roidb):  # for each image
        for cls_ind, cls in enumerate(json_dataset.classes):  # for each cls
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break

            gt_classes = roidb[i]['gt_classes']
            ind = np.where(gt_classes == cls_ind)
            if len(ind[0]) > 0:  # the image has the class
                gt_all[i, cls_ind] = 1
                dets = all_boxes[cls_ind][i]
                scores = dets[:, -1]
                if len(scores) > 0:
                    scores_cls[cls_ind].append(scores.max())
                else:  # no detection at all
                    scores_cls[cls_ind].append(0.0)

    for cls_ind, cls in enumerate(json_dataset.classes):  # for each cls
        if cls == '__background__':
            continue
        # sort scores for each class in ascending order
        tmp = sorted(scores_cls[cls_ind])
        pos = round(len(tmp) * 0.1)  # corresponding to recall of 0.9
        Th_cls[cls_ind] = tmp[pos]  # get the threshold for each cls

    for i, entry in enumerate(roidb):  # for each image
        for cls_ind, cls in enumerate(json_dataset.classes):  # for each cls
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break

            gt_classes = roidb[i]['gt_classes']
            ind = np.where(gt_classes == cls_ind)
            if len(ind[0]) > 0:
                gt_all[i, cls_ind] = 1

            dets = all_boxes[cls_ind][i]
            scores = dets[:, -1]
            if len(scores) > 0:
                pred_all[i, cls_ind] = scores.max()
            else:  # not detection
                pred_all[i, cls_ind] = 0

            # compute class level recall and FP
            if pred_all[i, cls_ind] > Th_cls[cls_ind]:
                if gt_all[i, cls_ind] == 1:
                    hit[cls_ind] += 1
                if gt_all[i, cls_ind] == 0:
                    FP[cls_ind] += 1

    # compute AUC
    print('AUC')
    AUROCs = compute_AUCs(gt_all, pred_all, json_dataset.classes)
    for c in range(1, num_classes):
        print(json_dataset.classes[c], "%.3f" % AUROCs[c])
    print("recall on class level")
    for c in range(1, num_classes):
        print("%.3f" % (hit[c] / gt_all[:, c].sum()))
    print("FP on class level")
    for c in range(1, num_classes):
        print("%.3f" % (FP[c] / (len(roidb) - gt_all[:, c].sum())))

    print(Th_cls)


def evaluate_class_AUC_orig(json_dataset, roidb, all_boxes, output_dir, cleanup=False):
    """ LJY """
    res_file = os.path.join(output_dir, 'bbox_' + json_dataset.name + '_results')
    res_file += '.json'

    num_classes = len(json_dataset.classes)

    gt_all = np.zeros((len(roidb), num_classes), dtype=np.uint8)
    pred_all = np.zeros((len(roidb), num_classes), dtype=np.float32)

    hit = [0 for _ in range(num_classes)]
    FP = [0 for _ in range(num_classes)]

    for i, entry in enumerate(roidb):  # for each image
        for cls_ind, cls in enumerate(json_dataset.classes):  # for each cls
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break

            gt_classes = roidb[i]['gt_classes']
            ind = np.where(gt_classes == cls_ind)
            if len(ind[0]) > 0:
                gt_all[i, cls_ind] = 1

            dets = all_boxes[cls_ind][i]
            scores = dets[:, -1]
            if len(scores) > 0:
                pred_all[i, cls_ind] = scores.max()
            else:  # not detection
                pred_all[i, cls_ind] = 0

            # compute class level recall and FP
            if pred_all[i, cls_ind] > 0.4:
                if gt_all[i, cls_ind] == 1:
                    hit[cls_ind] += 1
                if gt_all[i, cls_ind] == 0:
                    FP[cls_ind] += 1

    # compute AUC
    AUROCs = compute_AUCs(gt_all, pred_all, json_dataset.classes)
    for c in range(1, num_classes):
        print("%.3f" % AUROCs[c])
    print("recall on class level")
    for c in range(1, num_classes):
        print("%.3f" % (hit[c] / gt_all[:, c].sum()))
    print("FP on class level")
    for c in range(1, num_classes):
        print("%.3f" % (FP[c] / (len(roidb) - gt_all[:, c].sum())))


def _write_coco_bbox_results_file(json_dataset, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):  # for each class
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        # cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_bbox_results_one_category(json_dataset, all_boxes[cls_ind], cls_ind))
    logger.info('Writing bbox results json to: {}'.format(os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _coco_bbox_results_one_category(json_dataset, boxes, cat_id):
    results = []
    # image_ids = json_dataset.COCO.getImgIds()
    # image_ids.sort()
    # assert len(boxes) == len(image_ids)
    for i, dets in enumerate(boxes):
        dets = boxes[i]
        if isinstance(dets, list) and len(dets) == 0:
            continue
        dets = dets.astype(np.float)
        scores = dets[:, -1]
        xywh_dets = box_utils.xyxy_to_xywh(dets[:, 0:4])
        xs = xywh_dets[:, 0]
        ys = xywh_dets[:, 1]
        ws = xywh_dets[:, 2]
        hs = xywh_dets[:, 3]
        results.extend(
            [{'image_id': i,  # dummy, to be compatible with COCO
              'category_id': cat_id,
              'bbox': [xs[k], ys[k], ws[k], hs[k]],
              'score': scores[k]} for k in range(dets.shape[0])])
    return results


# def _coco_bbox_results_one_category(json_dataset, boxes, cat_id):
#     results = []
#     image_ids = json_dataset.COCO.getImgIds()
#     image_ids.sort()
#     assert len(boxes) == len(image_ids)
#     for i, image_id in enumerate(image_ids):
#         dets = boxes[i]
#         if isinstance(dets, list) and len(dets) == 0:
#             continue
#         dets = dets.astype(np.float)
#         scores = dets[:, -1]
#         xywh_dets = box_utils.xyxy_to_xywh(dets[:, 0:4])
#         xs = xywh_dets[:, 0]
#         ys = xywh_dets[:, 1]
#         ws = xywh_dets[:, 2]
#         hs = xywh_dets[:, 3]
#         results.extend(
#             [{'image_id': image_id,
#               'category_id': cat_id,
#               'bbox': [xs[k], ys[k], ws[k], hs[k]],
#               'score': scores[k]} for k in range(dets.shape[0])])
#     return results


def _do_detection_eval(json_dataset, res_file, output_dir):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'detection_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    return coco_eval


def _log_detection_eval_metrics(json_dataset, coco_eval):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    logger.info(
        '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
            IoU_lo_thresh, IoU_hi_thresh))
    logger.info('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        logger.info('{:.1f}'.format(100 * ap))
    logger.info('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()


def evaluate_box_proposals(json_dataset, roidb, thresholds=None, area='all', limit=None):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        'all': 0,
        'small': 1,
        'medium': 2,
        'large': 3,
        '96-128': 4,
        '128-256': 5,
        '256-512': 6,
        '512-inf': 7}
    area_ranges = [
        [0**2, 1e5**2],    # all
        [0**2, 32**2],     # small
        [32**2, 96**2],    # medium
        [96**2, 1e5**2],   # large
        [96**2, 128**2],   # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2]]  # 512-inf
    assert area in areas, 'Unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    num_pos = 0
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_boxes = entry['boxes'][gt_inds, :]
        gt_areas = entry['seg_areas'][gt_inds]
        valid_gt_inds = np.where(
            (gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
        gt_boxes = gt_boxes[valid_gt_inds, :]
        num_pos += len(valid_gt_inds)
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        boxes = entry['boxes'][non_gt_inds, :]
        if boxes.shape[0] == 0:
            continue
        if limit is not None and boxes.shape[0] > limit:
            boxes = boxes[:limit, :]
        overlaps = box_utils.bbox_overlaps(
            boxes.astype(dtype=np.float32, copy=False),
            gt_boxes.astype(dtype=np.float32, copy=False))
        _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        for j in range(min(boxes.shape[0], gt_boxes.shape[0])):
            # find which proposal box maximally covers each gt box
            argmax_overlaps = overlaps.argmax(axis=0)
            # and get the iou amount of coverage for each gt box
            max_overlaps = overlaps.max(axis=0)
            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ind = max_overlaps.argmax()
            gt_ovr = max_overlaps.max()
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        # append recorded iou coverage level
        gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    gt_overlaps = np.sort(gt_overlaps)
    if thresholds is None:
        step = 0.05
        thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
            'gt_overlaps': gt_overlaps, 'num_pos': num_pos}


def evaluate_keypoints(
    json_dataset,
    all_boxes,
    all_keypoints,
    output_dir,
    use_salt=True,
    cleanup=False
):
    res_file = os.path.join(
        output_dir, 'keypoints_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_keypoint_results_file(
        json_dataset, all_boxes, all_keypoints, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if json_dataset.name.find('test') == -1:
        coco_eval = _do_keypoint_eval(json_dataset, res_file, output_dir)
    else:
        coco_eval = None
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


def _write_coco_keypoint_results_file(json_dataset, all_boxes, all_keypoints, res_file):
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_keypoints):
            break
        logger.info(
            'Collecting {} results ({:d}/{:d})'.format(
                cls, cls_ind, len(all_keypoints) - 1))
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_kp_results_one_category(
            json_dataset, all_boxes[cls_ind], all_keypoints[cls_ind], cat_id))
    logger.info(
        'Writing keypoint results json to: {}'.format(
            os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _coco_kp_results_one_category(json_dataset, boxes, kps, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(kps) == len(image_ids)
    assert len(boxes) == len(image_ids)
    use_box_score = False
    if cfg.KRCNN.KEYPOINT_CONFIDENCE == 'logit':
        # This is ugly; see utils.keypoints.heatmap_to_keypoints for the magic indexes
        score_index = 2
    elif cfg.KRCNN.KEYPOINT_CONFIDENCE == 'prob':
        score_index = 3
    elif cfg.KRCNN.KEYPOINT_CONFIDENCE == 'bbox':
        use_box_score = True
    else:
        raise ValueError(
            'KRCNN.KEYPOINT_CONFIDENCE must be "logit", "prob", or "bbox"')
    for i, image_id in enumerate(image_ids):
        if len(boxes[i]) == 0:
            continue
        kps_dets = kps[i]
        scores = boxes[i][:, -1].astype(np.float)
        if len(kps_dets) == 0:
            continue
        for j in range(len(kps_dets)):
            xy = []

            kps_score = 0
            for k in range(kps_dets[j].shape[1]):
                xy.append(float(kps_dets[j][0, k]))
                xy.append(float(kps_dets[j][1, k]))
                xy.append(1)
                if not use_box_score:
                    kps_score += kps_dets[j][score_index, k]

            if use_box_score:
                kps_score = scores[j]
            else:
                kps_score /= kps_dets[j].shape[1]

            results.extend([{'image_id': image_id,
                             'category_id': cat_id,
                             'keypoints': xy,
                             'score': kps_score}])
    return results


def _do_keypoint_eval(json_dataset, res_file, output_dir):
    ann_type = 'keypoints'
    imgIds = json_dataset.COCO.getImgIds()
    imgIds.sort()
    coco_dt = json_dataset.COCO.loadRes(res_file)
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, ann_type)
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    eval_file = os.path.join(output_dir, 'keypoint_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    coco_eval.summarize()
    return coco_eval
