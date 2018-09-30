import mxnet as mx
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

#x = mx.nd.random.uniform(shape=(1,3,224,224))
x = mx.sym.var(name='data')



net = CTPNHead()
#net.collect_params().initialize()

z = net(x)
print(z)
#print(z.shape)
