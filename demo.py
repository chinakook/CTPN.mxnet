# -*- coding: utf-8 -*-

import numpy as np
import cv2
import mxnet as mx
from rcnn.io.image import resize, transform
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
from rcnn.text_connector.detectors import TextDetector
import matplotlib.pyplot as plt

from rcnn.symbol.proposal import ProposalOperator



class CTPNDetector(object):
    SCALES = (1000, 1600)
    PIXEL_MEANS = np.array([0.406, 0.456, 0.485]) # BGR order
    PIXEL_STDS = np.array([0.225, 0.224, 0.229]) # BGR order
    DATA_NAMES = ['data', 'im_info']


    def __init__(self, prefix, epoch):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        self.mod = mx.mod.Module(symbol=sym, context=[mx.gpu(0)], data_names=self.DATA_NAMES, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 600, 1000)), ('im_info', (1, 3))], label_shapes=None, force_rebind=False)
        self.mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)

        self.textdet = TextDetector()

    def _generate_batch(self, im):
        short_side = self.SCALES[0]
        long_side = self.SCALES[1]
        im_array, im_scale = resize(im, short_side, long_side)
        im_array = transform(im_array, self.PIXEL_MEANS, self.PIXEL_STDS)
        im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
        data = [mx.nd.array(im_array), mx.nd.array(im_info)]
        data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
        data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
        return data_batch, im_scale

    def det(self, raw_img):
        data_batch, im_scale = self._generate_batch(raw_img)
        self.mod.forward(data_batch)

        im_shape = data_batch.data[0].shape

        output = dict(zip(self.mod.output_names, self.mod.get_outputs()))

        rois = output['rois_output'].asnumpy()[:, 1:]
        scores = output['rois_score'].asnumpy()

        pred_boxes = clip_boxes(rois, im_shape[-2:])

        pred_boxes = pred_boxes / im_scale

        textrois = self.textdet.detect(pred_boxes, scores, (im.shape[0], im.shape[1]))

        return textrois


class OCR(object):
    CHARSET = b'./model/charset.txt'
    def __init__(self, prefix, epoch):
        with open(self.CHARSET) as fchar:
            self.charset = [_.decode('utf-8').strip() for _ in fchar.readlines()]
            #self.charset = [_.decode('gb18030').strip() for _ in fchar.readlines()]
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        self.mod = mx.mod.Module(symbol=sym, context=[mx.gpu(0)], data_names=['data'], label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 1, 32, 280))], label_shapes=None, force_rebind=False)
        self.mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)

    def get_string(self, label_list):
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
            s += c.encode('utf8')
        return s

    def rec(self, img):
        #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_w = int((img2.shape[0] / float(32)) * img2.shape[1])
        img2 = cv2.resize(img2, (norm_w, 32))

        #img2 = np.transpose(img2, axes=(2,0,1))
        img2 = np.expand_dims(img2, 0)
        img2 = np.expand_dims(img2, 0)
        # img2 = (img2 - 127.5) * 0.0078125
        batch = mx.io.DataBatch([mx.nd.array(img2)])
        self.mod.forward(batch)
        pred = self.mod.get_outputs()[0]
        prob = mx.nd.softmax(pred)
        am = mx.nd.argmax(prob, axis=-1)
        am_np = am.asnumpy()
        am_np = np.squeeze(am_np)
        label_list = list(am_np.astype(np.int32))
        s = self.get_string(label_list)
        return s 

if __name__ == '__main__':
    detector = CTPNDetector('./model/rpn1_deploy', 0)
    recog = OCR('./model/fp6', 0)
    im = cv2.imread('/mnt/15F1B72E1A7798FD/DK2/mpout/JPEGImages/IMG_20180929_162455.jpg')

    textrois = detector.det(im)

    for bbox in textrois:
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[5]
        roi = im[int(y0):int(y1), int(x0):int(x1), :]
        s = recog.rec(roi)
        print(s)
        plt.imshow(roi)
        plt.waitforbuttonpress()
        #cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)),
        #    (0,0,255), 1)

    plt.imshow(im[:,:,::-1])
    plt.waitforbuttonpress()

        