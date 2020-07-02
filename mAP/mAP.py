# author: lianjie
# created: 2019.9.9
# 1.与当前检测代码结合使用（检测结果以txt形式保存）。
# 2.评估模式分为两类：（1）iou模式（2）center模式（判断pred中心是否在gt内）
# 3.直接在类初始化函数中，改变相应路径和模式即可。
# 4.最后会生成一个results.txt文件，保存一些必要的结果。
# 注：
# 新版机制(核心思想-降低fp): 当两个pred对应一个gt时，tp = 2, fp = 0, recall = 1/1 precision = 1 / 1 + 0
# （1）tp会增加 但新增加的tp不会参与到recall和precision的计算中 （2）fp 减少 提高0.2fp下的recall
# 原版机制：当两个pred对应一个gt时，tp = 1, fp = 1, recall = 1/1 precision = 1 / 1 + 1

import os
import sys
import glob
import json
import pdb
import shutil
import math

class mAP():
    def __init__(self):
        self.mode = 'iou'  # 分为iou模式和center模式
        #self.mode = 'center'
        self.MINOVERLAP = 0.5       # iou模式的阈值
        self.fp_thre = 0.1 # fix the fp to 0.1
        self.gt_files = 'ground-truth/*.txt'
        # self.gt_files = '/home/lianjie/mmdetection/mAP/ground-truth/*.txt'
        self.pred_files = 'predicted/*.txt'
        self.files_num = len(glob.glob(self.gt_files))  # 2482
        self.tmp_files_path = "tmp_files/"
        if not os.path.exists(self.tmp_files_path): os.makedirs(self.tmp_files_path)
        self.results_file_path = "dcn_yy_nms0.5/"
        if not os.path.exists(self.results_file_path): os.makedirs(self.results_file_path)
        self.results_file = self.results_file_path + 'temp.txt'# 'base_gk_yy_0.25.txt'

        self.gt_counter_per_class, self.gt_classes = self.get_gt_info()
        self.pred_counter_per_class, self.pred_classes = self.get_pred_info()
        self.count_true_positives = {}

    def error(self, msg):
        print(msg)
        sys.exit(0)

    def file_lines_to_list(self, path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]

        return content

    """
     Ground-Truth
       Load each of the ground-truth files into a temporary ".json" file.
       Create a list of all the class names present in the ground-truth (gt_classes).
    """

    def get_gt_info(self):
        gt_files_list = glob.glob(self.gt_files)

        if len(gt_files_list) == 0:
            self.error("Error: No ground-truth files found!")
        gt_files_list.sort()
        gt_counter_per_class = {}

        for txt_file in gt_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # check if there is a correspondent predicted objects file
            if not os.path.exists('predicted/' + file_id + ".txt"):
                error_msg = "Error. File not found: predicted/" + file_id + ".txt\n"
                error(error_msg)

            lines_list = self.file_lines_to_list(txt_file)
            # create ground-truth dictionary
            bounding_boxes = []
            for line in lines_list:
                try:
                    class_name, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    self.error(error_msg)
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # count current object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1
            # dump bounding_boxes into a ".json" file
            with open(self.tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)

        return gt_counter_per_class, gt_classes

    """
    Predicted
    Load each of the predicted files into a temporary ".json" file.
    """

    def get_pred_info(self):
        pred_files_list = glob.glob(self.pred_files)

        if len(pred_files_list) == 0:
            self.error("Error: No ground-truth files found!")
        pred_files_list.sort()
        pred_counter_per_class = {}

        for class_index, class_name in enumerate(self.gt_classes):
            bounding_boxes = []
            for txt_file in pred_files_list:
                # the first time it checks if all the corresponding ground-truth files exist
                file_id = txt_file.split(".txt", 1)[0]
                file_id = os.path.basename(os.path.normpath(file_id))
                if class_index == 0:
                    if not os.path.exists('ground-truth/' + file_id + ".txt"):
                        error_msg = "Error. File not found: ground-truth/" + file_id + ".txt\n"
                        self.error(error_msg)
                lines = self.file_lines_to_list(txt_file)
                for line in lines:
                    try:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                        self.error(error_msg)
                    if tmp_class_name == class_name:
                        if class_name in pred_counter_per_class:
                            pred_counter_per_class[class_name] += 1
                        else:
                            pred_counter_per_class[class_name] = 1
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
            # sort predictions by decreasing confidence
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
            with open(self.tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        pred_classes = list(pred_counter_per_class.keys())
        pred_classes = sorted(pred_classes)

        return pred_counter_per_class, pred_classes

    def voc_ap(self, rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
          (goes from the end to the beginning)
          matlab:  for i=numel(mpre)-1:-1:1
                      mpre(i)=max(mpre(i),mpre(i+1));
        """

        # matlab indexes start in 1 but python in 0, so I have to do:
        #   range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #   range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
         This part creates a list of indexes where the recall changes
          matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
          (numerical integration)
          matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])

        return ap, mrec, mpre

    # 使用iou作为衡量标准，判断tp与fp
    def get_tp_fp_byiou(self, gt_file, bb, class_name):
        ground_truth_data = json.load(open(gt_file))

        tp = 0
        fp = 0

        # 一个pred可能与当前图片多个gt match 取ovmax最大的作为match的gt
        ovmax = -1
        gt_match = -1
        for obj in ground_truth_data:
            # look for a class_name match
            if obj["class_name"] == class_name:
                bbgt = [float(x) for x in obj["bbox"].split()]

                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                         (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        # 字典赋值 改变字典 变量均发生变化
                        gt_match = obj

                        # set minimum overlap
        min_overlap = self.MINOVERLAP
        if ovmax >= min_overlap:
            # 若有多个pred对应一个gt 则均作为当前gt的tp 但计算recall和precision时 以gt为基准
            if not bool(gt_match["used"]):
                # true positive
                tp = 1
                gt_match["used"] = True
                # pdb.set_trace()
                self.count_true_positives[class_name] += 1
                # update the ".json" file
                with open(gt_file, 'w') as f:
                    f.write(json.dumps(ground_truth_data))
            else:
                # tp 实际加1 但不参与到recall和precision的计算中 与原版相比: tp+1 fp-1
                self.count_true_positives[class_name] += 1
        else:
            fp = 1

        return tp, fp

        # 使用pred中心是否在gt内，判断tp与fp

    def get_tp_fp_bycenter(self, gt_file, bb, class_name):
        ground_truth_data = json.load(open(gt_file))

        tp = 0
        fp = 0

        dismin = sys.maxsize  # 一个pred的中心可能在多个gt框内 取中心距离最近的作为match的gt
        gt_match = -1
        center_pred = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]  # (x, y)
        for obj in ground_truth_data:
            # look for a class_name match
            if obj["class_name"] == class_name:
                bbgt = [float(x) for x in obj["bbox"].split()]
                center_gt = [(bbgt[0] + bbgt[2]) / 2, (bbgt[1] + bbgt[3]) / 2]
                # pdb.set_trace()
                if bbgt[0] < center_pred[0] < bbgt[2] and bbgt[1] < center_pred[1] < bbgt[3]:  # 判断pred中心点是否在gt矩形内
                    dis = math.sqrt((center_pred[0] - center_gt[0]) ** 2 + (center_pred[1] - center_gt[1]) ** 2)
                    if dis < dismin:
                        dismin = dis
                        gt_match = obj

        if gt_match != -1:
            if not bool(gt_match["used"]):  # 若有对个pred对应一个gt 只取score最高的pred作为当前gt的tp
                # true positive
                tp = 1
                gt_match["used"] = True
                self.count_true_positives[class_name] += 1
                # update the ".json" file
                with open(gt_file, 'w') as f:
                    f.write(json.dumps(ground_truth_data))
            else:
                # tp 实际加1 但不参与到recall和precision的计算中 与原版相比: tp+1 fp-1
                self.count_true_positives[class_name] += 1
        else:
            fp = 1

        return tp, fp

    '''
    Calculate the AP for each class: 分为两种方式 （1）传统的mAP计算方式 （2）pred的中心是否在gt内 
    '''

    def calc_AP(self):
        sum_AP = 0.0
        # open file to store the results
        with open(self.results_file, 'w') as results_file:
            # results_file.write("# AP and precision/recall per class\n")
            results_file.write("# AP per class\n")
            results_file.write('gt:标注框数量  pred:预测框数量  tp:以gt为基准的tp数量  ' + \
                               'ac_tp:实际检测的tp数量（多pred对一gt） fp:假阳数量' + '\n' + \
                               '注： pred = ac_tp + fp  ac_tp >= tp \n')
            recall_text = '\n' + '# recall: fix fp to the 0.2'
            # count_true_positives = {}
            for class_index, class_name in enumerate(self.gt_classes):
                self.count_true_positives[class_name] = 0
                """
                Load predictions of that class
                """
                predictions_file = self.tmp_files_path + "/" + class_name + "_predictions.json"
                predictions_data = json.load(open(predictions_file))
                """
                Assign predictions to ground truth objects
                """
                pred_num = len(predictions_data)
                tp = [0] * pred_num  # creates an array of zeros of size pred_num
                fp = [0] * pred_num
                for idx, prediction in enumerate(predictions_data):  # 按confidence降序
                    file_id = prediction["file_id"]
                    # assign prediction to ground truth object if any open ground-truth with that file_id
                    gt_file = self.tmp_files_path + "/" + file_id + "_ground_truth.json"
                    # load prediction bounding-box
                    bb = [float(x) for x in prediction["bbox"].split()]
                    # pdb.set_trace()
                    if self.mode == 'iou':
                        tp[idx], fp[idx] = self.get_tp_fp_byiou(gt_file, bb, class_name)
                    elif self.mode == 'center':
                        tp[idx], fp[idx] = self.get_tp_fp_bycenter(gt_file, bb, class_name)

                # compute precision/recall
                # pdb.set_trace()
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val

                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val

                # 计算 recall 以gt为基准
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / self.gt_counter_per_class[class_name]  # recall = tp / all_p

                # 计算 precsion 以gt为基准
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])  # precision = tp / tp + fp

                orig_rec, orig_prec = rec.copy(), prec.copy()

                # fix the fp to 0.2, first find where
                found = False
                for idx, _ in enumerate(fp):  # 显示
                    if _ >= self.files_num * self.fp_thre:
                        # pdb.set_trace()
                        score = float(predictions_data[_]['confidence'])
                        print(class_name + '  score:%.3f' % score)
                        recall_text += '\n' + class_name + '  score:%.3f' % score

                        print("%.3f" % orig_rec[idx])
                        print("%.3f" % (fp[idx] / self.files_num))
                        recall_text += '\n' + '%.3f' % orig_rec[idx]
                        recall_text += '\n' + '%.3f' % (fp[idx] / self.files_num)


                        found = True
                        break

                if not found:
                    if len(predictions_data) > 0:
                        score = float(predictions_data[-1]['confidence'])
                        print(class_name + '  score:%.3f' % score)
                        recall_text += '\n' + class_name + '  score:%.3f' % score
                    else:
                        print()
                    if len(orig_rec) > 0:
                        print("%.3f" % orig_rec[-1])
                        recall_text += '\n' + '%.3f' % orig_rec[-1]
                    else:
                        print()
                    if len(fp) > 0:
                        print("%.3f" % (fp[-1] / self.files_num))
                        recall_text += '\n' + '%.3f' % (fp[-1] / self.files_num)
                    else:
                        print()

                ap, mrec, mprec = self.voc_ap(rec, prec)
                sum_AP += ap
                text = '\n' + class_name + " AP = " + "{0:.2f}%".format(ap * 100)
                # gt:真实框数量 pred:预测框数量 tp:以gt为基准的tp ac_tp:实际的tp数量（多pred对一gt的情况）
                # fp:假阳数量     注： pred = ac_tp + fp  ac_tp >= tp
                text += '\n' + 'gt:%d ' % self.gt_counter_per_class[class_name] + \
                        'pred:%d ' % self.pred_counter_per_class[class_name] + \
                        'tp:%d ' % tp[-1] + 'ac_tp:%d ' % self.count_true_positives[class_name] + \
                        'fp:%d' % fp[-1] + '\n'
                results_file.write(text)
                """
                 Write to results.txt 
                """
                # text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
                # rounded_prec = ['%.2f' % elem for elem in prec]
                # rounded_rec = ['%.2f' % elem for elem in rec]
                # results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            results_file.write("\n\n# mAP of all classes\n")
            mAP = sum_AP / len(self.gt_classes)
            text = "mAP = {0:.2f}%".format(mAP * 100)
            results_file.write(text + "\n")
            results_file.write(recall_text + '\n')
            print(text)
            shutil.rmtree(self.tmp_files_path)


if __name__ == '__main__':
    evaluator = mAP()
    evaluator.calc_AP()