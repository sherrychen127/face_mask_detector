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
from facemask_dataset import register_facemask_dataset, get_facemask_1_dicts
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging

import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='pass argument to train model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=str, default='./output/facemask_train')
    parser.add_argument('--plot_only', action='store_true')
    parser.add_argument('--max_iter', type=int, default=2000)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    return args


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        # print(metrics_dict)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # print(merics)
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

class MyTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


#modify the config file
def modify_cfg(args, cfg_filepath = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_filepath))
    cfg.DATASETS.TRAIN = ("facemask_1_train",)
    cfg.DATASETS.TEST = ("facemask_1_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_filepath)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.max_iter    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = args.output

    cfg.TEST.EVAL_PERIOD = 200
    return cfg


def plot_loss(cfg):
    experiment_folder = cfg.OUTPUT_DIR
    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
    # print(experiment_metrics)
    plt.plot([x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
            [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])

    plt.plot([x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
            [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.show()


#train model
def train_model(args, cfg):
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__=='__main__':
    args = parse_args()
    # _, _ = register_facemask_dataset()
    _, _ = register_facemask_dataset(split='val')
    
    cfg = modify_cfg(args)
    if args.plot_only:
        plot_loss(cfg)
    else:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        train_model(args, cfg)
