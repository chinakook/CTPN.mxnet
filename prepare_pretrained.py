import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import rnn
from mxnet.gluon.model_zoo import vision

backbone = vision.vgg16(pretrained=True)

x = mx.nd.random.uniform(shape=(1,3,224,224))
backbone.hybridize()
y = backbone(x)
backbone.export('./model/vgggluon', 0)

backbone = vision.resnet101_v1(pretrained=True)
#print(backbone.features[:7])
x = mx.nd.random.uniform(shape=(1,3,224,224))
backbone.hybridize()
y = backbone(x)
backbone.export('./model/resnet101_v1', 0)
