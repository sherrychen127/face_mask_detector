import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from facemask_dataset import register_facemask_dataset, get_facemask_1_dicts
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='pass argument to train model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=str, default='./output/facemask_train')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    return args


#modify the config file
def modify_cfg(args, cfg_filepath = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_filepath))
    cfg.DATASETS.TRAIN = ("facemask_1_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_filepath)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = args.output
    return cfg


#train model
def train_model(args, cfg):
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
    trainer.train()



if __name__=='__main__':
    args = parse_args()
    facemask_1_metadata, dataset_dicts = register_facemask_dataset()
    cfg = modify_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    train_model(args, cfg)