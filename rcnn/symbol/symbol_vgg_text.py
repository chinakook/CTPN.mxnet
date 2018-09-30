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

import mxnet as mx
#from mxnet import rnn
from rcnn.config import config
from . import proposal
from . import proposal_target

from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import rnn
from mxnet.gluon.model_zoo import vision

class CTPNHead(gluon.HybridBlock):
    def __init__(self):
        super(CTPNHead, self).__init__()
        backbone = vision.vgg16(pretrained=True, root='./model')
        with self.name_scope():
            self.detect_head = backbone.features[:30]
            self.rpn_conv = nn.Conv2D(512, (3,3), strides=(1,1), padding=(1,1))
            self.biRnn = rnn.LSTM(128, 1, 'NTC', dropout=0.5, bidirectional=True)
            self.proj = nn.Dense(512, flatten=False)
    def hybrid_forward(self, F, x, **kwargs):
        x = self.detect_head(x)
        x = self.rpn_conv(x)
        x_t0 = F.transpose(x, axes=(0, 2, 3, 1))
        x_t = F.reshape(x_t0, shape=(-3, -2))
        lstm_o = self.biRnn(x_t)
        pred = self.proj(lstm_o)
        pred = F.reshape_like(pred, x_t0)
        pred = F.transpose(pred, axes=(0, 3, 1, 2))
        out = F.Activation(pred, act_type="relu")
        return out

def get_vgg_text_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    WRKLIMIT = 2048

    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=WRKLIMIT, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=WRKLIMIT, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=WRKLIMIT, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=WRKLIMIT, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=WRKLIMIT, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=WRKLIMIT, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=WRKLIMIT, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=WRKLIMIT, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=WRKLIMIT, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=WRKLIMIT, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=WRKLIMIT, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=WRKLIMIT, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=WRKLIMIT, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")

    return relu5_3

def get_vgg_ctpn_head(data):
    net = CTPNHead()

    return net(data)

def get_vgg_text_rpn(data, label, bbox_target, bbox_weight, num_anchors=10):
    """
    Region Proposal Network with VGG
    :param num_anchors: used to determine output size
    :return: Symbol
    """

    rpn_relu = get_vgg_ctpn_head(data)
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=label, multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1, name="cls_prob")
    # bounding box regression
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss])
    return group


def get_vgg_text_rpn_test(data, im_info, num_anchors=10):
    """
    Region Proposal Network with VGG
    :param num_anchors: used to determine output size
    :return: Symbol
    """

    rpn_relu = get_vgg_ctpn_head(data)
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        rois = mx.symbol.contrib.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.PROPOSAL_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.PROPOSAL_POST_NMS_TOP_N,
            threshold=config.TEST.PROPOSAL_NMS_THRESH, rpn_min_size=config.TEST.PROPOSAL_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.PROPOSAL_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.PROPOSAL_POST_NMS_TOP_N,
            threshold=config.TEST.PROPOSAL_NMS_THRESH, rpn_min_size=config.TEST.PROPOSAL_MIN_SIZE)
    # rois = group[0]
    # score = group[1]

    group = rois #mx.sym.Group([rois, rpn_cls_prob_reshape, rpn_bbox_pred])

    return group

