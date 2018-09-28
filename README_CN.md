# 基于MXNet的CTPN

## 简介
CTPN是一个不错的场景文字检测算法。

[**English Document**](./README.md)

## 训练
1. 编译Cython模块，如下:
    ``` bash
    cd rcnn/cython
    python setup.py build_ext --inplace
    cd ../pycocotools/
    python setup.py build_ext --inplace
    ```
2. 从 [google drive](https://drive.google.com/open?id=0B_WmJoEtfGhDRl82b1dJTjB2ZGc) 或 [baidu yun](https://pan.baidu.com/s/1kUNTl1l) 下载数据集. 数据集已由 **@eragonruan** 为CTPN专门准备好；
3. 解压下载好的数据集至 ```'VOCdevkit'``` 文件夹, 将 ```rcnn/config.py``` 中的 ```default.root_path``` 和 ```default.dataset_path``` 设置为 ```<你的目录>/VOCdevkit/VOC2007'```. 你也可以修改 ```rcnn/config.py``` 文件中的其他超参数；
4. 运行 ```python train_ctpn.py``` 可开始训练. 运行 ```python train_ctpn.py --gpus '0' --rpn_lr 0.01 --no_flip 0``` 在 gpu 0 上以学习率 0.01 和翻转数据增强模式进行训练。

## 测试
输入 ```python demo_ctpn.py --image "<你的图像路径>" --prefix model/rpn1 --epoch 8``` 进行测试。

## 我们的结果
`注意:` 以下全部图像均下载自互联网，如果对你有影响，请联系我进行删除。
<img src="/results/demo.jpg" width=320 height=240 /><img src="/results/demo2.jpg" width=320 height=240 />
<img src="/results/demo3.jpg" width=320 height=240 /><img src="/results/demo4.jpg" width=320 height=240 />
<img src="/results/demo5.jpg" width=320 height=240 />
<img src="/results/demo6.jpg" width=320 height=480 />
<img src="/results/demo7.jpg" width=480 height=320/>
<img src="/results/demo8.jpg" width=320 height=480/><img src="/results/demo9.jpg" width=320 height=480/>
<img src="/results/demo10.jpg" />

## 硬件需求
任何至少具备 **2GB** 显存的 NVIDIA 显卡都OK.

## 参考
1. https://github.com/tianzhi0549/CTPN
2. https://github.com/eragonruan/text-detection-ctpn

## TODO
- [ ] 准备自定义数据集的教程
- [ ] 支持Windows
- [ ] 支持网络模型的发布以及支持C++的推断