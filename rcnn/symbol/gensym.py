import mxnet as mx
from ..config import config
from . import *

symfun = eval('get_' + 'vgg_text' + '_rpn')

def gen_sym(data_shape_dict, ctxlen):
    s0 = data_shape_dict["data"]
    s1 = data_shape_dict["label"]
    s2 = data_shape_dict["bbox_target"]
    s3 = data_shape_dict["bbox_weight"]
    data = mx.symbol.Variable(name="data", shape=(s0[0]//ctxlen, s0[1], s0[2], s0[3]))
    label = mx.symbol.Variable(name='label', shape=(s1[0]//ctxlen, s1[1]))
    bbox_target = mx.symbol.Variable(name='bbox_target', shape=(s2[0]//ctxlen, s2[1], s2[2], s2[3]))
    bbox_weight = mx.symbol.Variable(name='bbox_weight', shape=(s3[0]//ctxlen, s3[1], s3[2], s3[3]))
    return symfun(data, label, bbox_target, bbox_weight, config.NUM_ANCHORS)

def gen_sym_infer(data_shape_dict, ctxlen):
    s0 = data_shape_dict["data"]
    s1 = data_shape_dict["im_info"]
    data = mx.symbol.Variable(name="data", shape=(s0[0]//ctxlen, s0[1], s0[2], s0[3]))
    im_info = mx.symbol.Variable(name='im_info', shape=(s1[0]//ctxlen, s1[1]))
    return get_vgg_text_rpn_test(data, im_info)