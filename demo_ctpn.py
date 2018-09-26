# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import os
import cv2
import mxnet as mx
import numpy as np
from rcnn.logger import logger
from rcnn.config import config
from rcnn.symbol import get_vgg_text_rpn_test, gensym
from rcnn.io.image import resize, transform
from rcnn.core.tester import Predictor, im_detect, im_rpn_detect, im_proposal, vis_all_detection, draw_all_detection
from rcnn.utils.load_model import load_param

from rcnn.text_connector.detectors import TextDetector

import matplotlib.pyplot as plt
import random
import logging

SHORT_SIDE = config.SCALES[0][0]
LONG_SIDE = config.SCALES[0][1]
PIXEL_MEANS = config.PIXEL_MEANS
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1, 3, SHORT_SIDE, LONG_SIDE)), ('im_info', (1, 3))]
LABEL_SHAPES = None
# visualization
CONF_THRESH = 0.7
NMS_THRESH = 0.3

def get_net(prefix, epoch, ctx):
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    predictor = Predictor(gensym.gen_sym_infer, DATA_NAMES, LABEL_NAMES, context=ctx, max_data_shapes= dict(DATA_SHAPES),
                          provide_data=DATA_SHAPES, provide_label=LABEL_SHAPES,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor


def generate_batch(im):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    data_names: names in data_batch
    im_scale: float number
    """
    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
    data = [mx.nd.array(im_array), mx.nd.array(im_info)]
    data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch, DATA_NAMES, im_scale

def demo_net(predictor, detector, image_name):
    """
    generate data_batch -> im_detect -> post process
    :param predictor: Predictor
    :param image_name: image name
    :param vis: will save as a new image if not visualized
    :return: None
    """
    assert os.path.exists(image_name), image_name + ' not found'
    im = cv2.imread(image_name)
    data_batch, data_names, im_scale = generate_batch(im)
    scores, boxes, data_dict = im_rpn_detect(predictor, data_batch, data_names, im_scale)

    textrois = detector.detect(boxes, scores, (im.shape[0], im.shape[1]))

    #plt.imshow(im[:,:,::-1])
    for bbox in textrois:
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[5]
        # color = (random.random(), random.random(), random.random())
        color = (0,1,0)
        
        rect = plt.Rectangle((x0, y0),
                                x1 - x0,
                                y1 - y0, fill=False,
                                edgecolor=color, linewidth=1.0)
        #plt.gca().add_patch(rect)        
        cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)),
            (0,0,255), 2)
    im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
    #cv2.imwrite('results/demo.jpg', im)
    cv2.imshow("w", im)
    cv2.waitKey()
    #plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network')
    parser.add_argument('--image', help='custom image', type=str)
    parser.add_argument('--prefix', help='saved model prefix', type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model', type=int)
    parser.add_argument('--gpu', help='GPU device to use', default=0, type=int)
    parser.add_argument('--vis', help='display result', action='store_true')
    args = parser.parse_args()
    return args


def main():

    #logger.setLevel(logging.DEBUG)

    args = parse_args()
    ctx = mx.gpu(args.gpu)

    predictor = get_net(args.prefix, args.epoch, ctx)

    detector = TextDetector()
    
    #path = '/home/dingkou/dev/text-detection-ctpn/data/demo'
    path = '/mnt/6B133E147DED759E/VOCdevkit/VOC2007/train/JPEGImages'
    flist = [path + '/' + fn for fn in os.listdir(path)]
    for fn in flist:
        pass
    demo_net(predictor, detector, '/mnt/15F1B72E1A7798FD/DK2/mop/png/Pic_2018_09_12_100837_blockId#5984.png')


if __name__ == '__main__':
    main()

    # python demo_ctpn.py --image "/mnt/15F1B72E1A7798FD/DK2/ctpn_tiny/val/JPEGImages/img_1765.jpg" --prefix model/rpn1 --epoch 8 --vis