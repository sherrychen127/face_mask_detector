
import detectron2
from detectron2.utils.logger import setup_logger


# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

from facemask_dataset import register_facemask_dataset, get_facemask_1_dicts

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("facemask_1_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.OUTPUT_DIR = './output/facemask_train' #EDIT


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)



# print(predictor.has('pred_mask'))

facemask_1_metadata = MetadataCatalog.get("facemask_1_train")

# im = cv2.imread('1200.jpg')
# outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
# print(outputs)
# print(outputs['instances'].has('pred_masks'))
# v = Visualizer(im[:, :, ::-1],
#                 metadata=facemask_1_metadata, 
#                 scale=0.5, 
#                 instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
# )
# out = v.draw_instance_predictions(outputs["instances"])#.to("cpu"))
# # cv2_imshow(out.get_image()[:, :, ::-1])
# print(out)
# plt.imshow(out.get_image()[:, :, ::-1])


facemask_1_metadata = MetadataCatalog.get("facemask_1_train")
# dataset_dicts = DatasetCatalog.get("facemask_1_train")
dataset_dicts = get_facemask_1_dicts("/home/sherry/APS360/datasets/face_mask_dataset/Medical mask/Medical mask/Medical Mask")

for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print(outputs)
    # v = Visualizer(im[:, :, ::-1],
    #                metadata=facemask_1_metadata, 
    #                scale=0.5, 
    #                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    # )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # # cv2_imshow(out.get_image()[:, :, ::-1])
    # print(out)
    # plt.imshow(out.get_image()[:, :, ::-1])
plt.show()