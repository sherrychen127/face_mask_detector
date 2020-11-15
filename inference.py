
import numpy as np
import os, json, cv2, random
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from facemask_dataset import register_facemask_dataset, get_facemask_1_dicts

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


args = parse_args()
cfg = modify_cfg(args) #use the same config as training
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

facemask_1_metadata = MetadataCatalog.get("facemask_1_val")
dataset_dicts = get_facemask_1_dicts("/home/sherry/APS360/datasets/face_mask_dataset/Medical mask/Medical mask/Medical Mask", split='val')

#randomly select 5 images to visualize
for d in random.sample(dataset_dicts, 5):    
    im = cv2.imread(d["file_name"])
    import time
    start = time.time()
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print('used', time.time() - start, 'sec')
    v = newVisualizer(im[:, :, ::-1],
                   metadata=facemask_1_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.new_draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])
    print(out)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()