import mxnet as mx
from ..config import config
from . import *

symfun = eval('get_' + 'vgg_text' + '_rpn')

def gen_sym():
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_weight = mx.symbol.Variable(name='bbox_weight')
    return symfun(data, label, bbox_target, bbox_weight, config.NUM_ANCHORS)

def gen_sym_infer():
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name='im_info')
    return get_vgg_text_rpn_test(data, im_info)