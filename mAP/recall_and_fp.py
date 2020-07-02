# This script is used to compute recall and fp
import os
import glob
import pdb
import numpy as np
import cv2
import json
import os.path as osp

_GRAY = (218, 227, 218)
_GREEN = (18, 255, 15)
_BLUE = (255, 18, 18)
_WHITE = (255, 255, 255)


def vis_boxes_ljy(im, gt_boxes, pred_boxes, text=None):
    for i in range(gt_boxes.shape[0]):
        (x0, y0, x1, y1) = gt_boxes[i]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(im, (x0, y0), (x1, y1), _GREEN, thickness=1)
    for i in range(pred_boxes.shape[0]):
        (x0, y0, x1, y1) = pred_boxes[i]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(im, (x0, y0), (x1, y1), _BLUE, thickness=1)

    cv2.putText(im, text, (50, 100), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 204, 0), thickness=2)
    return im


def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.
    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes
    Note:
        List format: [[xc, yc, w, h],...]
    """
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


types = ['高密度影', '结节钙化乳头影', '肋骨骨折异常']
# 531 beta 30
classes = ['肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '主动脉结增宽', '膈面异常', '结节',
           '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '乳头影', '弥漫性结节', '肺不张', '多发结节',
           '心影增大', '脊柱侧弯', '纵隔变宽', '肺门增浓', '膈下游离气体', '肋骨异常', '肺结核',
           '皮下气肿', '主动脉钙化', '空洞', '液气胸', '肋骨缺失', '肩关节异常'] + \
          types

cls2type = {
    '肺实变': '高密度影',
    '胸腔积液': '高密度影',
    '肿块': '高密度影',
    '肺不张': '高密度影',
    '纤维化表现': '高密度影',
    '肺结核': '高密度影',
    '多发结节': '高密度影',
    '弥漫性结节': '高密度影',

    '结节': '结节钙化乳头影',
    '钙化': '结节钙化乳头影',
    '乳头影': '结节钙化乳头影',

    '肋骨骨折': '肋骨骨折异常',
    '肋骨异常': '肋骨骨折异常',
    '肋骨缺失': '肋骨骨折异常'
}

cls2th = {  # 产品
    '肺实变': 0.5,
    '纤维化表现': 0.4,
    '胸腔积液': 0.4,
    '胸膜增厚': 0.4,
    '主动脉结增宽': 0.85,
    '膈面异常': 0.6,
    '结节': 0.2,
    '肿块': 0.3,
    '异物': 0.7,
    '气胸': 0.3,
    '肺气肿': 0.5,
    '骨折': 0.2,
    '钙化': 0.2,
    '乳头影': 0.4,
    '弥漫性结节': 0.2,
    '肺不张': 0.3,
    '多发结节': 0.2,
    '心影增大': 0.8,
    '脊柱侧弯': 0.3,
    '纵隔变宽': 0.2,
    '肺门增浓': 0.2,
    '膈下游离气体': 0.05,
    '肋骨异常': 0.2,
    '肺结核': 0.2,
    '皮下气肿': 0.2,
    '主动脉钙化': 0.3,
    '空洞': 0.1,
    '液气胸': 0.1,
    '肋骨缺失': 0.1,
    '肩关节异常': 0.1
}

gt_files = glob.glob(os.path.join('./ground-truth', "*.txt"))
pred_files = glob.glob(os.path.join('./predicted', "*.txt"))

assert len(gt_files) == len(pred_files)

nGT, nTP, nFP = {c: 0 for c in classes}, {c: 0 for c in classes}, {c: 0 for c in classes}


anno = json.load(open('/data1/DX/test_anno_DX.json'))
for e in anno:
    # pdb.set_trace()
    evaId = e['evaId']
    docName = e['docName']
    if docName not in ['Zhang_ZW', 'fj6311', 'yangyan123', 'liuxiaohong']:
        docName = 'others'
    print(evaId)
    gt_file = osp.join('ground-truth', e['filename'] + '.txt')
    pred_file = osp.join('predicted', e['filename'] + '.txt')
    img_file = e['filename'] + '.jpg'

    im = cv2.imread(osp.join('/data1/DX/fold_all', img_file))

    preds, gts = {c: [] for c in classes}, {c: [] for c in classes}  # e.g. {'骨折': [[x1,y1,x2,y2], ...]}

    unique_boxes = {}  # remove duplicate boxes assigned to different classes, keep largest-score class
    for line in open(pred_file):
        tmp = line.strip().split()
        cls, score, x1, y1, x2, y2 = tmp
        score = float(score)
        if score < cls2th[cls]:
            continue
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        hash_key = ''.join(str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2))
        if hash_key in unique_boxes:
            if score > unique_boxes[hash_key][1]:
                unique_boxes[hash_key] = [cls, score, x1, y1, x2, y2]
        else:
            unique_boxes[hash_key] = [cls, score, x1, y1, x2, y2]

    for k, v in unique_boxes.items():
        cls = v[0]
        preds[cls].append(v[2:])
        if cls in cls2type:  # add type
            preds[cls2type[cls]].append(v[2:])

    for k, v in preds.items():
        preds[k] = np.array(preds[k])

    for line in open(gt_file):
        tmp = line.strip().split()
        cls, x1, y1, x2, y2 = tmp
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        gts[cls].append([x1, y1, x2, y2])
        if cls == '弥漫性结节' or cls == '多发结节':  # 多发结节的片子预测到结节也算对
            gts['结节'].append([x1, y1, x2, y2])

        if cls in cls2type:  # add type
            gts[cls2type[cls]].append([x1, y1, x2, y2])

    for k, v in gts.items():
        gts[k] = np.array(gts[k])

    # pdb.set_trace()
    for cls, gt_boxes in gts.items():  # for each class
        pred_boxes = preds[cls]

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        elif len(gt_boxes) > 0 and len(pred_boxes) == 0:  # FN
            nGT[cls] += gt_boxes.shape[0]
            # if cls == '结节':
            #     im = vis_boxes_ljy(im, gt_boxes, pred_boxes, text=str(evaId) + ' ' + docName)
            #     cv2.imwrite(osp.join('vis', docName, img_file), im)

        elif len(gt_boxes) == 0 and len(pred_boxes) > 0:  # FP
            nFP[cls] += pred_boxes.shape[0]
            # if cls == '结节':
            #     im = vis_boxes_ljy(im, gt_boxes, pred_boxes, text=str(evaId) + ' ' + docName)
            #     cv2.imwrite(osp.join('vis', docName, img_file), im)

        elif len(gt_boxes) > 0 and len(pred_boxes) > 0:
            nGT[cls] += gt_boxes.shape[0]
            IOUs = bbox_ious(pred_boxes, gt_boxes)

            nTP[cls] += np.sum(np.max(IOUs, axis=0) > 0)  # 看真值有多少hit
            nFP[cls] += pred_boxes.shape[0] - np.sum(np.max(IOUs, axis=1) > 0)  # 看预测有多少empty

            # if cls == '结节':
            #     im = vis_boxes_ljy(im, gt_boxes, pred_boxes, text=str(evaId) + ' ' + docName)
            #     cv2.imwrite(osp.join('vis', docName, img_file), im)


sum_FP = 0
N = len(gt_files)
print('N ', N)
for c, n in nGT.items():
    if c not in types:
        sum_FP += nFP[c]
    else:
        print('--------------------------------------------------')
    if n > 0:
        print('{} {:.3f} {:.3f}'.format(c, nTP[c] / n, nFP[c] / N))
    else:
        print('{} {:.3f} {:.3f}'.format(c, 0, nFP[c] / N))

    # print("{%s, %.3f, %.3f}".format(c, nTP[c] / n, nFP[c] / 2443))


print(sum_FP / N)
