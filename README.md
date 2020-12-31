# face_mask_detector
Real-time (~14 fps) face mask detector which accurately identifies bounding boxes of faces and labels whether a mask is worn (correctly) or not worn using a Faster-RCNN (with FPN backbone) architecture. Achieves 85% mAP@0.5 on the kaggle dataset: https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset.

## Getting Started
### Environment
1. Python>=3.6  
2. Based on the runtime environment, i.e. CPU, CUDA GPU, please install the following packages:  
    - Install Detectron2>=0.2, please refer to its official [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
    - Install PyTorch>=1.4, please refer to its official website [INSTALL](https://pytorch.org)
3. Install additional packages
    ```
    pip install -r requirements.txt
    ```
### Prepare Dataset
Create symlinks of the downloaded datasets to `./datasets`     
```
cd ./datasets
ln -sfn /path/to/download/dataset1 dataset1
```
