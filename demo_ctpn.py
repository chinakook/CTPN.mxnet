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
from rcnn.symbol import get_vgg_text_rpn_test
from rcnn.io.image import resize, transform
from rcnn.core.tester import Predictor, im_detect, im_rpn_detect, im_proposal, vis_all_detection, draw_all_detection
from rcnn.utils.load_model import load_param
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

from rcnn.text_connector import text_proposal_connector

import matplotlib.pyplot as plt
import random
import logging

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
config.TEST.HAS_RPN = True
SHORT_SIDE = config.SCALES[0][0]
LONG_SIDE = config.SCALES[0][1]
PIXEL_MEANS = config.PIXEL_MEANS
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1, 3, 600, 1000)), ('im_info', (1, 3))]
LABEL_SHAPES = None
# visualization
CONF_THRESH = 0.7
NMS_THRESH = 0.3
nms = py_nms_wrapper(NMS_THRESH)

def gen_sym_infer(data_shape_dict, ctxlen):
    s0 = data_shape_dict["data"]
    s1 = data_shape_dict["im_info"]
    data = mx.symbol.Variable(name="data", shape=(s0[0]//ctxlen, s0[1], s0[2], s0[3]))
    im_info = mx.symbol.Variable(name='im_info', shape=(s1[0]//ctxlen, s1[1]))
    return get_vgg_text_rpn_test(data, im_info)

def get_net(prefix, epoch, ctx):
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    # infer shape
    data_shape_dict = dict(DATA_SHAPES)

    symbol = gen_sym_infer(data_shape_dict, len(ctx) if isinstance(ctx, list) else 1)
    # data = mx.symbol.Variable(name="data", shape=(1,3,600,903))
    # im_info = mx.symbol.Variable(name="im_info", shape=(1,3))
    # symbol = get_vgg_text_rpn_test(data, im_info)

    arg_names, aux_names = symbol.list_arguments(), symbol.list_auxiliary_states()
    arg_shape, _, aux_shape = symbol.infer_shape_partial()
    arg_shape_dict = dict(zip(arg_names, arg_shape))
    aux_shape_dict = dict(zip(aux_names, aux_shape))

    # check shapes
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    predictor = Predictor(gen_sym_infer, DATA_NAMES, LABEL_NAMES, context=ctx, max_data_shapes=data_shape_dict,
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


def filter_boxes(boxes):
    heights=np.zeros((len(boxes), 1), np.float)
    widths=np.zeros((len(boxes), 1), np.float)
    scores=np.zeros((len(boxes), 1), np.float)
    index=0
    for box in boxes:
        heights[index]=(abs(box[5]-box[1])+abs(box[7]-box[3]))/2.0+1
        widths[index]=(abs(box[2]-box[0])+abs(box[6]-box[4]))/2.0+1
        scores[index] = box[8]
        index += 1
    MIN_RATIO=0.5
    LINE_MIN_SCORE=0.9
    TEXT_PROPOSALS_WIDTH=16
    MIN_NUM_PROPOSALS = 2
    return np.where((widths/heights>MIN_RATIO) & (scores>LINE_MIN_SCORE) &
                        (widths>(TEXT_PROPOSALS_WIDTH*MIN_NUM_PROPOSALS)))[0]

def demo_net(predictor, image_name, vis=False):
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

    keep = np.where(scores >= CONF_THRESH)[0]

    #sorted_indices = np.argsort(scores.ravel())[::-1]
    #boxes, scores = boxes[sorted_indices], scores[sorted_indices]

    


    dets = np.hstack((boxes, scores)).astype(np.float32)[keep, :]
    keep = nms(dets)

    boxes, scores = boxes[keep], scores[keep]

    tpc = text_proposal_connector.TextProposalConnector()
    im_size = (im.shape[0], im.shape[1])
    text_rects = tpc.get_text_lines(boxes, scores, im_size)
    keep_inds = filter_boxes(text_rects)
    print(text_rects[keep_inds])

    all_boxes = text_rects[keep_inds] #dets[keep, :]

    #print(boxes)
    plt.imshow(im)
    for bbox in all_boxes:
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[5]
        color = (random.random(), random.random(), random.random())
        
        rect = plt.Rectangle((x0, y0),
                                x1 - x0,
                                y1 - y0, fill=False,
                                edgecolor=color, linewidth=1.0)
        plt.gca().add_patch(rect)        
        # cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
        #     (0,255,0), 1)
    #cv2.imshow("w", im)
    #cv2.waitKey()
    plt.show()
    # all_boxes = [[] for _ in CLASSES]
    # for cls in CLASSES:
    #     cls_ind = CLASSES.index(cls)
    #     cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    #     # cls_scores = scores[:, cls_ind, np.newaxis]
    #     # keep = np.where(cls_scores >= CONF_THRESH)[0]
    #     # dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
    #     # keep = nms(dets)
    #     all_boxes[cls_ind] = cls_boxes #dets[keep, :]

    # boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]

    # # print results
    # logger.info('---class---')
    # logger.info('[[x1, x2, y1, y2, confidence]]')
    # for ind, boxes in enumerate(boxes_this_image):
    #     if len(boxes) > 0:
    #         logger.info('---%s---' % CLASSES[ind])
    #         logger.info('%s' % boxes)

    # if vis:
    #     vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
    # else:
    #     result_file = image_name.replace('.', '_result.')
    #     logger.info('results saved to %s' % result_file)
    #     im = draw_all_detection(data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
    #     cv2.imwrite(result_file, im)


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

    logger.setLevel(logging.DEBUG)

    args = parse_args()
    ctx = mx.gpu(args.gpu)

    predictor = get_net(args.prefix, args.epoch, ctx)
    demo_net(predictor, args.image, args.vis)


if __name__ == '__main__':
    main()

    # python demo_ctpn.py --image "/mnt/15F1B72E1A7798FD/DK2/ctpn_tiny/val/JPEGImages/img_1765.jpg" --prefix model/rpn1 --epoch 8 --vis