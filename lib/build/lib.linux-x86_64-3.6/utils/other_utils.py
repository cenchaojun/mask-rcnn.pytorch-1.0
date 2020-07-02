import torch
from sklearn.metrics import roc_auc_score
import cv2
import numpy as np
import pdb

def compute_AUCs(real, pred, classes):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    for i in range(len(classes)):
        one_cls_real_np = real[:, i]
        one_cls_pred_np = pred[:, i]
        score = 0
        try:
            score = roc_auc_score(one_cls_real_np, one_cls_pred_np)
        except Exception as e:
            # print(e)
            pass
        AUROCs.append(score)

    return AUROCs


# numpy version
def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) * (gt_boxes[:,3] - gt_boxes[:,1] + 1)).reshape(1, K)
    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) * (anchors[:,3] - anchors[:,1] + 1)).reshape(N, 1)

    # pdb.set_trace()
    boxes = np.expand_dims(anchors, axis=1).repeat(K, axis=1)  # Nx4 -> Nx1x4 -> NxKx4
    # boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = np.expand_dims(gt_boxes, axis=0).repeat(N, axis=0)  # Kx4 -> 1xkx4 -> NxKx4
    # query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (np.minimum(boxes[:,:,2], query_boxes[:,:,2]) - np.maximum(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (np.minimum(boxes[:,:,3], query_boxes[:,:,3]) - np.maximum(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def vis_detections(im, class_name, gt_boxes, dets, thresh=0.8):
    """Visual debugging of detections."""
    if dets is not None:
        for i in range(dets.shape[0]):
            bbox = tuple(int(np.round(x)) for x in dets[i, :4])
            if dets.shape[1] >= 5:
                score = dets[i, -1]
                cv2.putText(im, '%.3f' % score, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 204, 0), thickness=1)

            cv2.rectangle(im, bbox[0:2], bbox[2:4], (255, 0, 0), 3)

            # if score > thresh:
            #     # cv2.putText(im, '%s: %.3f' % (class_names[i], score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
            #     #            1.0, (255, 0, 0), thickness=1)
            #     cv2.putText(im, '%.3f' % score, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
            #                 1.0, (0, 204, 0), thickness=1)
    if gt_boxes is not None:
        cv2.putText(im, '%s' % class_name, (15, 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=1)
        for i in range(gt_boxes.shape[0]):
            bbox = tuple(int(np.round(x)) for x in gt_boxes[i])
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 255, 0), 1)

    return im