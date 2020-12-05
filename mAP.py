# #!/usr/bin/python
# from __future__ import division
# from __future__ import print_function
import cv2
import os
import numpy as np
import time
import pickle
import json

import numpy as np
import os, json, cv2, random
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, _create_text_labels, ColorMode
from inference import newVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import matplotlib.pyplot as plt
from facemask_dataset import register_facemask_dataset, get_facemask_1_dicts

from train import parse_args, modify_cfg

args = parse_args()
cfg = modify_cfg(args) #use the same config as training
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.WEIGHTS = '/home/salar/RVL/aps/face_mask_detector/checkpoints/fasterrcnn_model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

predictor = DefaultPredictor(cfg)

# facemask_1_metadata = MetadataCatalog.get("facemask_1_val")
facemask_1_metadata, dataset_dicts = register_facemask_dataset(split='val')

vis = False

# gt_data = [0,0,0]
# count = 0
# for d in dataset_dicts:
#     if count % 100 == 0:
#         print(count, '/', len(dataset_dicts))
#     count += 1
#     im = cv2.imread(d["file_name"])

#     for element in d['annotations']:
#         gt_id = element['category_id']
#         gt_data[gt_id] += 1
# print('gt data', gt_data)
gt_class_data = [565, 517] #, 21] #0,1,2

MINOVERLAP = 0.5

def draw_bbox_d2(img, labels, out_path=None):
    d2_colors = {"with_mask": "c", "without_mask": "r", "mask_weared_incorrect": "orange"}
    height, width, _ = img.shape
    # fig = plt.figure()
    ax = plt.subplot(1, 2, 2)

    # ax = fig.add_axes([0, 0, 2, 2])
    ax.imshow(img)
    for label in labels:
        classname = label[0]
        bbox = label[1:]
#         print(bbox)
        ax.add_patch(plt.Rectangle(tuple(bbox[:2]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                   edgecolor=d2_colors.get(classname, "g"), linewidth=1, facecolor="None"))
        plt.text(bbox[0], bbox[1], classname, fontsize=15, color=d2_colors.get(classname, "g"))
    # plt.show()
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')

def box_IoU(boxA,boxB):
    IoU = -1
    #topleft and bottomright (x,y) coordinates of the intersection rectangle
    x1 = max(boxA[0],boxB[0])
    y1 = max(boxA[1],boxB[1])
    x2 = min(boxA[2],boxB[2])
    y2 = min(boxA[3],boxB[3])
    #width and height of the intersection rectangle
    iw = x2 - x1 + 1
    ih = y2 - y1 + 1
    #make sure width and height are both bigger than 0
    if iw > 0 and ih > 0:
        interArea = iw * ih
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        unionArea = boxA_area + boxB_area - interArea
        IoU = interArea / unionArea
    return IoU

def voc_ap(recall, precision):
    recall.insert(0, 0.0)
    recall.append(1.0)
    mrecall = recall[:]
    precision.insert(0, 0.0)
    precision.append(0.0)
    mprecision = precision[:]
    #Compute a version of the measured precision/recall curve with precision monotonically decreasing, by setting the precision for recall r to the maximum precision obtained for any recall r' > r
    #from pascal VOC2010 official website
    for i in range(len(mprecision)-2, -1, -1):
        mprecision[i] = max(mprecision[i], mprecision[i+1])

    #i_list save the index where the recall's value changes
    i_list = []
    for i in range(1, len(mrecall)):
        if mrecall[i] != mrecall[i-1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrecall[i]-mrecall[i-1]) * mprecision[i])
    return ap, mrecall, mprecision

dump = False
if dump:
    # tp_ptr = [0,0]
    # fp_ptr = [0,0]

    # correct = 0
    # total = 0
    count = 0
    # res = [[0,0], [0,0]]
    # file_list = ['./datasets/dataset1/Medical mask/Medical mask/Medical Mask/images/5525.jpg',
    #         './datasets/dataset1/Medical mask/Medical mask/Medical Mask/images/3067.png',
    #         './datasets/dataset1/Medical mask/Medical mask/Medical Mask/images/5028.jpg',
    #         './datasets/dataset1/Medical mask/Medical mask/Medical Mask/images/5907.jpg']

    # total_TP = [[[0,0]]*label_len for label_len in gt_data]
    # total_FP = [[[0,0]]*label_len for label_len in gt_data]
    # print(len(total_TP), len(total_TP[0]), len(total_TP[1]))

    pred = [[],[]]
    for d in dataset_dicts:
        if count % 100 == 0:
            print(count, '/', len(dataset_dicts))
        count += 1

        file_name = os.path.basename(d["file_name"])
        
        #ground truth
        gt = []
        for element in d['annotations']:
            gt_id = element['category_id']
            gt_box = element['bbox'].tolist()
            # gt.append([gt_id, gt_box, 0])
            gt.append({'label':gt_id, 'bbox': gt_box, 'used': False})

        with open("./ground_truth/" + file_name + "_gt.json", 'w') as out:
            json.dump(gt, out)

        #prediction
        im = cv2.imread(d["file_name"])
        outputs = predictor(im) 
        predictions = np.array(outputs["instances"].pred_classes.detach().cpu().tolist())
        # p_bbox = []
        for i in range(len(predictions)):
            pred_id = predictions[i]
            pred_bbox = outputs["instances"].pred_boxes.tensor.detach().cpu().tolist()[i]
            pred_score = outputs["instances"].scores.detach().cpu().tolist()[i]
            # pred[d["file_name"]] = [pred_id, pred_bbox, pred_score]
            pred[pred_id].append({"confidence":pred_score, "file_name":file_name, "bbox":pred_bbox})

    with open('./prediction/0_p.json', 'w') as out:
        pred[0].sort(key=lambda x:x['confidence'], reverse=True)
        json.dump(pred[0], out)

    with open('./prediction/1_p.json', 'w') as out:
        pred[1].sort(key=lambda x:x['confidence'], reverse=True)
        json.dump(pred[1], out)        


#clear used variable:
import glob
for gt_file in glob.glob('./ground_truth/*.json'):
    # gt_file = "./ground_truth/" + file_name + "_gt.json"
    # if not os.path.exists(gt_file):
    #     continue
    gt_data = json.load(open(gt_file))
    
    for gt_obj in gt_data:
        gt_obj['used']=False
    
    with open(gt_file, 'w') as f:
        f.write(json.dumps(gt_data))

#Calculate ap
sum_AP = 0.0
result_text = ""
total_recall = []
total_precision = []
for label in [0,1]:
    p_file = "./prediction/" + str(label) + "_p.json"
    p_data = json.load(open(p_file))

    tp = [0] * len(p_data)
    fp = [0] * len(p_data)

    for index, prediction_obj in enumerate(p_data):
        file_name = prediction_obj["file_name"].replace('.jpg', '.png')
        gt_file = "./ground_truth/" + file_name + "_gt.json"
        if not os.path.exists(gt_file):
            file_name = prediction_obj["file_name"]
            gt_file = "./ground_truth/" + file_name + "_gt.json"
            if not os.path.exists(gt_file):
                continue
        gt_data = json.load(open(gt_file))

        IoU_max = -1
        gt_match = -1

        p_bbox = [float(x) for x in prediction_obj["bbox"]]
        for gt_obj in gt_data:
            if gt_obj["label"] == label:
                gt_bbox = [float(x) for x in gt_obj["bbox"]]
                IoU = box_IoU(p_bbox,gt_bbox)
                if IoU > IoU_max:
                    IoU_max = IoU
                    gt_match = gt_obj

        if IoU_max > MINOVERLAP:
            if not bool(gt_match["used"]):
                tp[index] = 1
                gt_match["used"] = True
                with open(gt_file, 'w') as f:
                    f.write(json.dumps(gt_data))
            else:
                fp[index] = 1
        else:
            fp[index] = 1

    sum = 0
    for index, value in enumerate(tp):
        tp[index] += sum
        sum += value
    sum = 0
    for index, value in enumerate(fp):
        fp[index] += sum
        sum += value
    print(label, tp[-1], fp[-1])

    recall = tp[:]
    for index, value in enumerate(tp):
        recall[index] = float(tp[index]) / gt_class_data[label]
    precision = tp[:]
    for index, value in enumerate(tp):
        if tp[index] + fp[index] == 0:
            precision[index] = 1.0
            continue
        precision[index] = float(tp[index]) / (fp[index] + tp[index])
    total_recall.append(recall)
    total_precision.append(precision)
    ap, mrecall, mprecision = voc_ap(recall, precision)
    sum_AP += ap
    text = "{0:.2f}%  ".format(ap * 100) + str(label) + " AP"
    result_text += text + "\n"

gt_label = [0,1]
mAP = sum_AP / len(gt_label)
result_text += "{0:.2f}%  mAP".format(mAP * 100)
with open("./result.txt", 'w') as result:
    result.write(result_text)
print(result_text)


import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

plot_confusion_matrix = False
plot_pr_curve = True
if plot_confusion_matrix:
    array = [[504, 64], [62, 430]]
    array_normed = array / np.sum(array)
    print(array_normed)
    df_cm = pd.DataFrame(array)
    plt.figure(figsize = (7,5))
    labels = ['True Neg','False Pos','False Neg','True Pos']
    labels = np.asarray(labels).reshape(2,2)
    sn.heatmap(df_cm/np.sum(df_cm), annot=labels, fmt='', cmap='Blues')
    plt.show()


if plot_pr_curve:
    plt.figure(figsize = (7,5))
    # plt.plot(total_recall[0], total_precision[0])
    # plt.plot(total_recall[1], total_precision[1])
    plt.plot(mrecall, mprecision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('2-class Precision-Recall Curve @ 0.5')
    plt.show()

