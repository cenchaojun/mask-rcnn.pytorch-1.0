import glob
import json
import os
import shutil
import operator
import sys
import argparse
import pdb

MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
  args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
  specific_iou_flagged = True

# if there are no images then no animation can be shown
img_path = 'images'
if os.path.exists(img_path):
  for dirpath, dirnames, files in os.walk(img_path):
    if not files:
      # no image files found
      args.no_animation = True
else:
  args.no_animation = True

# try to import OpenCV if the user didn't choose the option --no-animation
show_animation = False
if not args.no_animation:
  try:
    import cv2
    show_animation = True
  except ImportError:
    print("\"opencv-python\" not found, please install to visualize the results.")
    args.no_animation = True

# try to import Matplotlib if the user didn't choose the option --no-plot
draw_plot = False
# if not args.no_plot:
#   try:
#     import matplotlib.pyplot as plt
#     draw_plot = True
#   except ImportError:
#     print("\"matplotlib\" not found, please install it to get the resulting plots.")
#     args.no_plot = True


#throw error and exit

def error(msg):
  print(msg)
  sys.exit(0)

'''
check if the number is a float between 0.0 and 1.0
'''

def is_float_between_0_and_1(value):
  try:
    val = float(value)
    if val > 0.0 and val < 1.0:
      return True
    else:
      return False
  except ValueError:
    return False

"""
 Calculate the AP given the recall and precision array
  1st) We compute a version of the measured precision/recall curve with
       precision monotonically decreasing
  2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
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
  rec.insert(0, 0.0) # insert 0.0 at begining of list
  rec.append(1.0) # insert 1.0 at end of list
  mrec = rec[:]
  prec.insert(0, 0.0) # insert 0.0 at begining of list
  prec.append(0.0) # insert 0.0 at end of list
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
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
  """
   This part creates a list of indexes where the recall changes
    matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  """
  i_list = []
  for i in range(1, len(mrec)):
    if mrec[i] != mrec[i-1]:
      i_list.append(i) # if it was matlab would be i + 1
  """
   The Average Precision (AP) is the area under the curve
    (numerical integration)
    matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  ap = 0.0
  for i in i_list:
    ap += ((mrec[i]-mrec[i-1]) * mpre[i])
  return ap, mrec, mpre

"""
Convert the lines of a file to a list
"""
def file_lines_to_list(path):
  # open txt file lines to a list
  with open(path) as f:
    content = f.readlines()
  # remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content]
  return content

"""
 Draws text in image
"""
def draw_text_in_image(img, text, pos, color, line_width):
  font = cv2.FONT_HERSHEY_PLAIN
  fontScale = 1
  lineType = 1
  bottomLeftCornerOfText = pos
  cv2.putText(img, text,
      bottomLeftCornerOfText,
      font,
      fontScale,
      color,
      lineType)
  text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
  return img, (line_width + text_width)


"""
 Create a "tmp_files/" and "results/" directory
"""
tmp_files_path = "tmp_files"
if not os.path.exists(tmp_files_path):  # if it doesn't exist already
  os.makedirs(tmp_files_path)
results_files_path = "results"

'''
if os.path.exists(results_files_path):  # if it exist already
  # reset the results directory
  shutil.rmtree(results_files_path)
os.makedirs(results_files_path)
'''

if draw_plot:
  os.makedirs(results_files_path + "/classes")
if show_animation:
  os.makedirs(results_files_path + "/images")
  os.makedirs(results_files_path + "/images/single_predictions")

"""
 Ground-Truth
   Load each of the ground-truth files into a temporary ".json" file.
   Create a list of all the class names present in the ground-truth (gt_classes).
"""

# get a list with the ground-truth files
ground_truth_files_list = glob.glob('ground-truth/*.txt')
if len(ground_truth_files_list) == 0:
  error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
# dictionary with counter per class
gt_counter_per_class = {}

for txt_file in ground_truth_files_list:
  #print(txt_file)
  file_id = txt_file.split(".txt", 1)[0]
  file_id = os.path.basename(os.path.normpath(file_id))
  # check if there is a correspondent predicted objects file
  if not os.path.exists('predicted/' + file_id + ".txt"):
    error_msg = "Error. File not found: predicted/" + file_id + ".txt\n"
    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
    error(error_msg)
  lines_list = file_lines_to_list(txt_file)
  # create ground-truth dictionary
  bounding_boxes = []
  is_difficult = False
  for line in lines_list:
    try:
      if "difficult" in line:
          class_name, left, top, right, bottom, _difficult = line.split()
          is_difficult = True
      else:
          class_name, left, top, right, bottom = line.split()
    except ValueError:
      error_msg = "Error: File " + txt_file + " in the wrong format.\n"
      error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
      error_msg += " Received: " + line
      error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
      error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
      error(error_msg)
    # check if class is in the ignore list, if yes skip
    if class_name in args.ignore:
      continue
    bbox = left + " " + top + " " + right + " " +bottom
    if is_difficult:
        bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
        is_difficult = False
    else:
        bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
        # count that object
        if class_name in gt_counter_per_class:
          gt_counter_per_class[class_name] += 1
        else:
          # if class didn't exist yet
          gt_counter_per_class[class_name] = 1
  # dump bounding_boxes into a ".json" file
  with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

gt_classes = list(gt_counter_per_class.keys())
# let's sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)
#print(gt_classes)
#print(gt_counter_per_class)

'''
 Check format of the flag --set-class-iou (if used)
  e.g. check if class exists
'''

if specific_iou_flagged:
  n_args = len(args.set_class_iou)
  error_msg = \
    '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
  if n_args % 2 != 0:
    error('Error, missing arguments. Flag usage:' + error_msg)
  # [class_1] [IoU_1] [class_2] [IoU_2]
  # specific_iou_classes = ['class_1', 'class_2']
  specific_iou_classes = args.set_class_iou[::2] # even
  # iou_list = ['IoU_1', 'IoU_2']
  iou_list = args.set_class_iou[1::2]  # odd
  if len(specific_iou_classes) != len(iou_list):
    error('Error, missing arguments. Flag usage:' + error_msg)
  for tmp_class in specific_iou_classes:
    if tmp_class not in gt_classes:
          error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
  for num in iou_list:
    if not is_float_between_0_and_1(num):
      error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 Predicted
   Load each of the predicted files into a temporary ".json" file.
"""

# get a list with the predicted files
predicted_files_list = glob.glob('predicted/*.txt')
predicted_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
  bounding_boxes = []
  for txt_file in predicted_files_list:
    #print(txt_file)
    # the first time it checks if all the corresponding ground-truth files exist
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    if class_index == 0:
      if not os.path.exists('ground-truth/' + file_id + ".txt"):
        error_msg = "Error. File not found: ground-truth/" + file_id + ".txt\n"
        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
        error(error_msg)
    lines = file_lines_to_list(txt_file)
    for line in lines:
      try:
        tmp_class_name, confidence, left, top, right, bottom = line.split()
      except ValueError:
        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
        error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
        error_msg += " Received: " + line
        error(error_msg)
      if tmp_class_name == class_name:
        #print("match")
        bbox = left + " " + top + " " + right + " " +bottom
        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        #print(bounding_boxes)
  # sort predictions by decreasing confidence
  bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
  with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

'''
 Calculate the AP for each class
'''
sum_AP = 0.0
ap_dictionary = {}
file_name = '/results_fpn.txt'
# open file to store the results
with open(results_files_path + file_name, 'w') as results_file:
  results_file.write("# AP and precision/recall per class\n")
  count_true_positives = {}
  for class_index, class_name in enumerate(gt_classes):
    count_true_positives[class_name] = 0
    """
     Load predictions of that class
    """
    predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
    predictions_data = json.load(open(predictions_file))

    """
     Assign predictions to ground truth objects
    """

    nd = len(predictions_data)
    tp = [0] * nd  # creates an array of zeros of size nd
    fp = [0] * nd
    for idx, prediction in enumerate(predictions_data):
      file_id = prediction["file_id"]
      # assign prediction to ground truth object if any open ground-truth with that file_id
      gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
      ground_truth_data = json.load(open(gt_file))
      ovmax = -1
      gt_match = -1
      # load prediction bounding-box
      bb = [float(x) for x in prediction["bbox"].split()]
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
              gt_match = obj

      # set minimum overlap
      min_overlap = MINOVERLAP
      if specific_iou_flagged:
        if class_name in specific_iou_classes:
          index = specific_iou_classes.index(class_name)
          min_overlap = float(iou_list[index])
      if ovmax >= min_overlap:
        if "difficult" not in gt_match:
            if not bool(gt_match["used"]):
              # true positive
              tp[idx] = 1
              gt_match["used"] = True
              count_true_positives[class_name] += 1
              # update the ".json" file
              with open(gt_file, 'w') as f:
                  f.write(json.dumps(ground_truth_data))
              if show_animation:
                status = "MATCH!"
            else:
              # false positive (multiple detection)
              fp[idx] = 1
              if show_animation:
                status = "REPEATED MATCH!"
      else:
        # false positive
        fp[idx] = 1
        if ovmax > 0:
          status = "INSUFFICIENT OVERLAP"

    # compute precision/recall
    # pdb.set_trace()
    cumsum = 0
    for idx, val in enumerate(fp):
      fp[idx] += cumsum
      cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
      tp[idx] += cumsum
      cumsum += val
    # pdb.set_trace()
    rec = tp[:]
    for idx, val in enumerate(tp):
      rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]

    prec = tp[:]
    for idx, val in enumerate(tp):
      prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    orig_rec, orig_prec = rec.copy(), prec.copy()

    # fix the fp to 0.2, first find where
    print(class_name)
    # pdb.set_trace()
    found = False
    for jdx, _ in enumerate(fp):
        if _ >= 2482 * 0.2:  # 2443, 1638 * 0.2 = 327.6, 371 * 0.2 = 74.2, 1876 * 0.2 = 375.2, 1876*4*0.2=1500.8
            print("%.3f" % orig_rec[jdx])
            print("%.3f" % (fp[jdx] / 2482))

            found = True
            break

    if not found:
        if len(orig_rec) > 0:
          print("%.3f" % orig_rec[-1])
        else:
          print()
        if len(fp) > 0:
          print("%.3f" % (fp[-1] / 2482))
        else:
          print()

    ap, mrec, mprec = voc_ap(rec, prec)  # LJY: 'mrec' seem to have 2 more zeros ahead of 'rec'
    sum_AP += ap
    text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP  "  # class_name + " AP = {0:.2f}%".format(ap*100)
    """
     Write to results.txt
    """
    # print(rec)
    # print(prec)
    # print(len(tp))
    # print(len(fp))
    # print(len(rec))
    # print(len(prec))

    # print(class_name)
    # print(class_name, ' recall ', "%.3f" % (orig_rec[-1]), ' precision ', "%.3f" % (prec[-2]), ' FP ', "%.3f" % (fp[-1] / 1642))

    rounded_prec = ['%.2f' % elem for elem in prec]
    rounded_rec = ['%.2f' % elem for elem in rec]
    results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

    # if not args.quiet:
    #   print(text)
    ap_dictionary[class_name] = ap

  results_file.write("\n# mAP of all classes\n")
  mAP = sum_AP / n_classes
  text = "mAP = {0:.2f}%".format(mAP * 100)
  results_file.write(text + "\n")
  print(text)

# remove the tmp_files directory
shutil.rmtree(tmp_files_path)

"""
 Count total of Predictions
"""
# iterate through all the files
pred_counter_per_class = {}
# all_classes_predicted_files = set([])
for txt_file in predicted_files_list:
  # get lines to list
  lines_list = file_lines_to_list(txt_file)
  for line in lines_list:
    class_name = line.split()[0]
    # check if class is in the ignore list, if yes skip
    if class_name in args.ignore:
      continue
    # count that object
    if class_name in pred_counter_per_class:
      pred_counter_per_class[class_name] += 1
    else:
      # if class didn't exist yet
      pred_counter_per_class[class_name] = 1
# print(pred_counter_per_class)
pred_classes = list(pred_counter_per_class.keys())

"""
 Write number of ground-truth objects per class to results.txt
"""
with open(results_files_path + file_name, 'a') as results_file:
  results_file.write("\n# Number of ground-truth objects per class\n")
  for class_name in sorted(gt_counter_per_class):
    results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

"""
 Finish counting true positives
"""
for class_name in pred_classes:
  # if class exists in predictions but not in ground-truth then there are no true positives in that class
  if class_name not in gt_classes:
    count_true_positives[class_name] = 0
# print(count_true_positives)

"""
 Write number of predicted objects per class to results.txt
"""
with open(results_files_path + file_name, 'a') as results_file:
  results_file.write("\n# Number of predicted objects per class\n")
  for class_name in sorted(pred_classes):
    n_pred = pred_counter_per_class[class_name]
    text = class_name + ": " + str(n_pred)
    text += " (tp:" + str(count_true_positives[class_name]) + ""
    text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
    results_file.write(text)