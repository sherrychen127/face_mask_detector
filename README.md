# face_mask_detector

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
