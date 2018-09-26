import numpy as np
import mxnet as mx

def rpn_nms(data, thresh, pre_nms_topN, post_nms_topN, ctx):
    src_pred = mx.nd.array(data, ctx)
    nms_pred = mx.nd.contrib.box_nms(src_pred, overlap_thresh=0.3, topk=pre_nms_topN
        , coord_start=0, score_index=4, id_index=-1, force_suppress=True)
    effect = int(mx.nd.sum(nms_pred[:, 0] >= 0).asscalar())

    if post_nms_topN < effect:
        nms_pred = nms_pred[:post_nms_topN, :]
    else:
        nms_pred = nms_pred[:effect, :]
        pad_size = post_nms_topN - effect

        pad = np.random.choice(effect, pad_size)
        idx = np.hstack((np.arange(effect), pad))

        nms_pred = mx.nd.take(nms_pred, mx.nd.array(idx, ctx))
    
    rpn_bbox = nms_pred[:, :4]
    rpn_scores = nms_pred[:, 4]
    rpn_scores = mx.nd.expand_dims(rpn_scores, axis=1)
    return rpn_bbox.asnumpy(), rpn_scores.asnumpy()