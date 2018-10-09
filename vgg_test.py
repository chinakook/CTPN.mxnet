import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import rnn
from mxnet.gluon.model_zoo import vision

backbone = vision.vgg16(pretrained=True, root='./model')

x = mx.nd.random.uniform(shape=(1,3,224,224))
backbone.hybridize()
y = backbone(x)
backbone.export('./model/vgggluon', 0)
