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
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# facemask_1_metadata = MetadataCatalog.get("facemask_1_val")
facemask_1_metadata, dataset_dicts = register_facemask_dataset(split='val')
args = parse_args()
cfg = modify_cfg(args) #use the same config as training
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

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
gt_data = [565, 517] #, 21] #0,1,2

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

pickle_dump = True
if pickle_dump:
    tp_ptr = [0,0]
    fp_ptr = [0,0]

    correct = 0
    total = 0
    count = 0
    res = [[0,0], [0,0]]
    file_list = ['./datasets/dataset1/Medical mask/Medical mask/Medical Mask/images/5525.jpg',
            './datasets/dataset1/Medical mask/Medical mask/Medical Mask/images/3067.png',
            './datasets/dataset1/Medical mask/Medical mask/Medical Mask/images/5028.jpg',
            './datasets/dataset1/Medical mask/Medical mask/Medical Mask/images/5907.jpg']

    total_TP = [[[0,0]]*label_len for label_len in gt_data]
    total_FP = [[[0,0]]*label_len for label_len in gt_data]
    print(len(total_TP), len(total_TP[0]), len(total_TP[1]))

    gt = {}
    # pred = {}
    pred = []
    for d in dataset_dicts:
        if count % 100 == 0:
            print(count, '/', len(dataset_dicts))
        count += 1
        
        # if d['file_name'] not in file_list:
        #     continue

        im = cv2.imread(d["file_name"])
        outputs = predictor(im) 
        predictions = np.array(outputs["instances"].pred_classes.detach().cpu().numpy())
        # pred = []
        # print(outputs["instances"].scores)
        p_bbox = []
        for i in range(len(predictions)):
            pred_id = predictions[i]
            pred_bbox = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()[i]
            pred_score = outputs["instances"].scores.detach().cpu().numpy()[i]
            # pred.append([pred_id, pred_bbox, pred_score])
            pred[d["file_name"]] = [pred_id, pred_bbox, pred_score]
        
        # gt = []
        for element in d['annotations']:
            gt_id = element['category_id']
            gt_box = element['bbox']
            # gt.append([gt_id, gt_box, 0])
            gt[d["file_name"]] = [gt_id, gt_box, 0]

    with open('gt.json', 'w') as f:
        json.dump(gt, f)
    with open('pred.json', 'w') as f:
        json.dump(pred, f)


#         for label in [0,1]:
#             # TP = 0
#             # FP = 0
#             TP = total_TP[label]
#             FP = total_FP[label]
#             for index, (pred_label, pred_bbox, pred_score) in enumerate(pred):
#                 if pred_label != label:
#                     continue
#                 IoU_max = -1
#                 gt_match = -1

#                 for gt_id, (gt_label, gt_bbox, used) in enumerate(gt):
#                     if gt_label == pred_label and gt_label == label:
#                         IoU = box_IoU(pred_bbox, gt_bbox)
#                         if IoU > IoU_max:
#                             IoU_max = IoU
#                             gt_match = [gt_id, gt_label, used]
                
#                 if IoU_max > MINOVERLAP:
#                     if not gt_match[2]: #not used
#                         gt[gt_match[0]][2] = 1 
#                         TP[index] = [1, pred_score]
#                         FP[index] = [0, pred_score]
#                     else:
#                         TP[index] = [0, pred_score]
#                         FP[index] = [1, pred_score]
#                 else:
#                     TP[index] = [0, pred_score]
#                     FP[index] = [1, pred_score]
#             total_TP[label] = TP
#             total_FP[label] = FP

#     with open('TP.pickle', 'wb') as f:
#         pickle.dump(total_TP, f)
#     with open('FP.pickle', 'wb') as f:
#         pickle.dump(total_FP, f)
#     print('finished dumping')

# else:
#     with open('TP.pickle', 'rb') as f:
#         total_TP = pickle.load(f)
#     with open('FP.pickle', 'rb') as f:
#         total_FP = pickle.load(f)
# # print(sum(total_TP[0][:][0]))
# for label in [0,1]:
#     TP = total_TP[label]
#     FP = total_FP[label]

#     TP.sort(key=lambda x:x[1], reverse=True)
#     FP.sort(key=lambda x:x[1], reverse=True)
#     # print(TP, FP)
#     #calculate precision - recall curve
#     sum = 0
#     for index, (value,_) in enumerate(TP):
#         TP[index][0] += sum
#         sum += value
#     sum = 0
#     for index, (value,_) in enumerate(TP):
#         FP[index][0] += sum
#         sum += value
#     recall = TP[:][0]
#     for index, (value,_) in enumerate(TP):
#         recall_index = float(TP[index][0])/gt_data[label]
#     precision = TP[:][0]
#     for index, (value,_) in enumerate(TP):
#         precision[index] = float(TP[index][0]) / (FP[index][0] + TP[index][0])

#     ap, mrecall, mprecision = voc_ap(recall, precision)
#     print(ap)



        # res.append((TP, FP))
        # res[label][0] += TP
        # res[label][1] += FP
        # print('label:', label, TP, FP)
        
        # if FP > 0 and vis:
        #     f = plt.figure(figsize=(11,5))
        #     ax1 = plt.subplot(1, 2, 1)
        #     print(d)
        #     v = newVisualizer(im[:, :, ::-1],
        #                    metadata=facemask_1_metadata, 
        #                    scale=0.5, 
        #                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        #     )
        #     out = v.new_draw_instance_predictions(outputs["instances"].to("cpu"))
        #     ax1.imshow(out.get_image())
        #     labels = []
        #     for r in gt:
        #         if r[0] == 0:
        #             cur_l = ['with_mask']
        #         else:
        #             cur_l = ['no_mask']
        #         cur_l.extend(list(r[1]))
        #         labels.append(cur_l)
        #     print(labels)
        #     draw_bbox_d2(im[:, :, ::-1], labels)
        #     # out = v.new_draw_instance_predictions(outputs["instances"].to("cpu"))
        #     plt.show()
print(res)
