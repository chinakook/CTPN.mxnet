# Connectionist Text Proposal Network in MXNet

## Introduction
CTPN is a nice scene text detection method.

[**中文文档**](./README_CN.md)

## Training
1. Build the cython modules as following:
    ``` bash
    cd rcnn/cython
    python setup.py build_ext --inplace
    cd ../pycocotools/
    python setup.py build_ext --inplace
    ```
2. Download dataset from [google drive](https://drive.google.com/open?id=0B_WmJoEtfGhDRl82b1dJTjB2ZGc) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l). This dataset is already prepared by **@eragonruan** to fit CTPN.
3. Unzip the dataset downloaded to ```'VOCdevkit'``` folder, and set both ```default.root_path``` and ```default.dataset_path``` in ```rcnn/config.py``` to ```'<somewhere>/VOCdevkit/VOC2007'```. You can also change other hyperparams in ```rcnn/config.py```.
4. Run ```python train_ctpn.py``` to train. Run ```python train_ctpn.py --gpus '0' --rpn_lr 0.01 --no_flip 0``` to train model on gpu 0 with learning rate 0.01 and with flip data augmentation.

## Testing
Use ```python demo_ctpn.py --image "<your_image_path>" --prefix model/rpn1 --epoch 8``` to test.

## Our results
`NOTICE:` all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="/results/demo.jpg" width=320 height=240 /><img src="/results/demo2.jpg" width=320 height=240 />
<img src="/results/demo3.jpg" width=320 height=240 /><img src="/results/demo4.jpg" width=320 height=240 />
<img src="/results/demo5.jpg" width=320 height=240 />
<img src="/results/demo6.jpg" width=320 height=480 />
<img src="/results/demo7.jpg" width=480 height=320/>
<img src="/results/demo8.jpg" width=320 height=480/><img src="/results/demo9.jpg" width=320 height=480/>
<img src="/results/demo10.jpg" />

## Requirements: Hardware
Any NVIDIA GPUs with at least **2GB** memory should be OK.

## References
1. https://github.com/tianzhi0549/CTPN
2. https://github.com/eragonruan/text-detection-ctpn

## TODO
- [ ] Custom dataset preparation tutorial
- [ ] Windows support
- [ ] Deploying network and c++ inference support