
import numpy as np
import os, json, cv2, random
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, _create_text_labels, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import matplotlib.pyplot as plt
from facemask_dataset import register_facemask_dataset, register_facemask_dataset_2

from train import parse_args, modify_cfg
CLASS_NAMES = ['face_with_mask', 'face_no_mask', 'face_with_mask_incorrect']

class newVisualizer(Visualizer):
    '''
    since Dectron's method does not work, use a wrapper to rewrite the draw_instance_prediction methods.
    '''
    def __init__(self, img_rgb, metadata=None, scale=None, instance_mode=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def new_draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, CLASS_NAMES)#self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        # if predictions.has("pred_masks"):
        #     masks = np.asarray(predictions.pred_masks)
        #     masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        # else:
        masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(None)
                # (predictions.pred_masks.any(dim=0) > 0).numpy()
                # if predictions.has("pred_masks")
                # else None
            
            alpha = 0.3
        print(labels)
        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

def get_retina_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.RETINANET.NUM_CLASSES = 3  # 3 class labels
    cfg.DATASETS.TRAIN = ("facemask_1_train",)
    cfg.DATASETS.TEST = ("facemask_1_val",)
    cfg.MODEL.WEIGHTS = "checkpoints/retinanet_final.pth"
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5   # set a custom testing
    return cfg

print('hi')

faster_r_cnn = True
if faster_r_cnn:
    args = parse_args()
    cfg = modify_cfg(args, ) #use the same config as training
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
else:
    cfg = get_retina_cfg()
predictor = DefaultPredictor(cfg)

# facemask_1_metadata = MetadataCatalog.get("facemask_1_val")
facemask_1_metadata, dataset_dicts = register_facemask_dataset(split='val')

# evaluator = COCOEvaluator("facemask_1_val", ("bbox", "segm"), False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "facemask_1_val")
# print(inference_on_dataset(predictor, val_loader, evaluator))
# # another equivalent way to evaluate the model is to use `trainer.test`

def print_class_acc(dataset_dicts):
    correct = 0
    total = 0
    count = 0
    for d in dataset_dicts:
        if count % 100 == 0:
            print(count, '/', len(dataset_dicts))
        count += 1
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        # print(outputs["instances"].pred_classes.detach().cpu().numpy())
        # print([dict['category_id'] for dict in d["annotations"]])   
        pred = np.array(outputs["instances"].pred_classes.detach().cpu().numpy())
        ground_truth = np.array([dict['category_id'] for dict in d["annotations"]])
        
        ml = min(len(pred), len(ground_truth))
        diff = pred[:ml] - ground_truth[:ml]
        correct += len(ground_truth) - len(np.where(diff>0)[0])
        total += len(ground_truth)
    print(correct, total)
    print('total class accuracy: ', correct/total)

#print_class_acc(dataset_dicts)

#randomly select 5 images to visualize
def visualize_predictions(facemask_metadata):
    for d in random.sample(dataset_dicts, 5):    
        im = cv2.imread(d["file_name"])
        import time
        start = time.time()
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        print('used', time.time() - start, 'sec')
        v = newVisualizer(im[:, :, ::-1],
                       metadata=facemask_metadata, 
                       scale=0.5, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.new_draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
        #print(out)
        #plt.imshow(out.get_image()[:, :, ::-1])
        #plt.show()

visualize_predictions(facemask_1_metadata)

#facemask_2_metadata, dataset2_dicts = register_facemask_dataset_2()
#visualize_predictions(facemask_2_metadata)

