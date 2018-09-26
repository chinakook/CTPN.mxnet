# Connectionist Text Proposal Network in MXNet

## Introduction
A MXNet CTPN implementation mimic of eragonruan's [tensorflow implementation](https://github.com/eragonruan/text-detection-ctpn) with full feature. We use eragonruan's dataset for training and get nearly the same detection results as his.

## Training
1. Download data prepared by eragonruan from [google drive](https://drive.google.com/open?id=0B_WmJoEtfGhDRl82b1dJTjB2ZGc) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l).
2. Unzip the dataset downloaded to 'VOCdevkit' folder, then move some folders and make the tree as following:
  VOCdevkit
     |
     |-VOC2007
        |- train
            |- Annotations
            |- JPEGImages
            |- Main (Moved from ImageSets folder)
                |- train.txt
3. Remove img_2346, img_2347, img_2348 and img_2349 from train.txt as these files dose not exist.
4. Change the Line 120 and 121 of config.py to the dataset folder.
5. run 'python train_ctpn.py'

## Testing
Use demo_ctpn.py to test.

## Our results
`NOTICE:` all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="/results/demo.jpg" width=320 height=240 />

## References
1. https://github.com/tianzhi0549/CTPN
2. https://github.com/eragonruan/text-detection-ctpn