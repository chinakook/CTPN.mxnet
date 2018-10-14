# -*- coding: utf-8 -*-
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import numpy as np
import cv2
import mxnet as mx
from mxnet import gluon
from rcnn.io.image import resize, transform
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
from rcnn.text_connector.detectors import TextDetector
import matplotlib.pyplot as plt

from rcnn.symbol.proposal import ProposalOperator

ctx = mx.gpu(0)

class CTPNDetector(object):
    SCALES = (1000, 1600)
    PIXEL_MEANS = np.array([0.406, 0.456, 0.485]) # BGR order
    PIXEL_STDS = np.array([0.225, 0.224, 0.229]) # BGR order

    def __init__(self, fn_symbol, fn_params):
        self.net = gluon.SymbolBlock.imports(fn_symbol, ['data', 'im_info'], fn_params, ctx=ctx)
        self.textdet = TextDetector()

    def _generate_batch(self, im):
        short_side = self.SCALES[0]
        long_side = self.SCALES[1]
        im_array, im_scale = resize(im, short_side, long_side)
        im_array = transform(im_array, self.PIXEL_MEANS, self.PIXEL_STDS)
        im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
        return mx.nd.array(im_array, ctx=ctx), mx.nd.array(im_info, ctx=ctx), im_scale

    def det(self, raw_img):
        im_array, im_info, im_scale = self._generate_batch(raw_img)
        output = self.net(im_array, im_info)

        im_shape = im_array.shape

        rois = output[0].asnumpy()[:, 1:]
        scores = output[1].asnumpy()

        pred_boxes = clip_boxes(rois, im_shape[-2:])

        pred_boxes = pred_boxes / im_scale

        textrois = self.textdet.detect(pred_boxes, scores, (raw_img.shape[0], raw_img.shape[1]))

        return textrois


class OCR(object):
    CHARSET = b'./model/charset.txt'
    def __init__(self, fn_symbol, fn_params):
        with open(self.CHARSET) as fchar:
            self.charset = [_.strip() for _ in fchar.readlines()]
            #self.charset = [_.decode('gb18030').strip() for _ in fchar.readlines()]
        self.net = gluon.SymbolBlock.imports(fn_symbol, ['data'], fn_params, ctx=ctx)

    def _get_string(self, label_list):
        # Do CTC label rule
        # CTC cannot emit a repeated symbol on consecutive timesteps
        ret = []
        label_list2 = [0] + list(label_list)
        for i in range(len(label_list)):
            c1 = label_list2[i]
            c2 = label_list2[i+1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        s = ''
        for l in ret:
            if l > 0 and l < len(self.charset):
                c = self.charset[l]
            else:
                c = ''
            s += c #.encode('utf8')
        return s

    def rec(self, img):
        #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_w = int((img2.shape[0] / float(32)) * img2.shape[1])
        img2 = cv2.resize(img2, (norm_w, 32))

        #img2 = np.transpose(img2, axes=(2,0,1))
        img2 = np.expand_dims(img2, 0)
        img2 = np.expand_dims(img2, 0)
        pred = self.net(mx.nd.array(img2, ctx=ctx))
        prob = mx.nd.softmax(pred)
        am = mx.nd.argmax(prob, axis=-1)
        am_np = am.asnumpy()
        am_np = np.squeeze(am_np)
        label_list = list(am_np.astype(np.int32))
        s = self._get_string(label_list)
        return s 

if __name__ == '__main__':

    from prepare_dataset.split_xml import write_xml

    detector = CTPNDetector('./model/rpn1_deploy-symbol.json', './model/rpn1_deploy-0000.params')
    recog = OCR('./model/fp6-symbol.json', './model/fp6-0000.params')
    
    def doocr(fn):
        im = cv2.imread(fn)
        textrois = detector.det(im)
        extras = []
        for bbox in textrois:
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[5]
            roi = im[int(y0):int(y1), int(x0):int(x1), :]
            s = recog.rec(roi)

            extras.append(s)

        write_xml(fn, im.shape[0], im.shape[1], textrois[:, (0,1,2,5)], None)

    testpath = '/home/kk/data/mp0'

    if os.path.isdir(testpath):
        flist = []
        for fn in os.listdir(testpath):
            if fn.endswith('.jpg'):
                flist.append(os.path.join(testpath, fn))
        for fn in flist:
            doocr(fn)
    else:
        doocr(fn)

        