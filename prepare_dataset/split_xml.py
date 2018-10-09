# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import cv2

from lxml import etree
from lxml.etree import Element, SubElement, tostring  

TARGET_SIZE = 1000
MAX_SIZE = 1600

def write_xml(img_name, height, width, bboxes, extra=None):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(img_name)
    node_path = SubElement(node_root, 'path')
    node_path.text = img_name
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for k in range(bboxes.shape[0]):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = "text"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bboxes[k][0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bboxes[k][1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(bboxes[k][2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bboxes[k][3])

        if extra is not None:
            node_extra = SubElement(node_object, 'extra')
            node_extra.text = extra[k]
    
    xml = tostring(node_root, pretty_print=True)
    xml = str(xml, encoding='utf-8')
    with open(img_name[:-3]+'xml','w') as f: ## Write document to file
        f.write(xml)

def splitXml(fn, out_img_path, out_xml_path, txt):
    tree = etree.parse(fn)

    imgfn = tree.xpath('//filename')[0].text

    xmldir = os.path.dirname(fn)

    img = cv2.imread(os.path.join(xmldir, imgfn))
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])
    im_scale = float(TARGET_SIZE) / float(im_size_min)
    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = float(MAX_SIZE) / float(im_size_max)
    re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    re_size = re_im.shape

    imgbasefn, ext = os.path.splitext(imgfn)
    cv2.imwrite(os.path.join(out_img_path, imgbasefn + '.jpg'), re_im)

    bboxes = np.empty(shape=(0, 4), dtype=np.int32)

    for bbox in tree.xpath('//bndbox'):
        xmin = int(bbox.getchildren()[0].text)
        ymin = int(bbox.getchildren()[1].text)
        xmax = int(bbox.getchildren()[2].text)
        ymax = int(bbox.getchildren()[3].text)

        xmin = int(float(xmin) / img_size[1] * re_size[1])
        ymin = int(float(ymin) / img_size[0] * re_size[0])
        xmax = int(float(xmax) / img_size[1] * re_size[1])
        ymax = int(float(ymax) / img_size[0] * re_size[0])

        if xmin < 0:
            xmin = 0
        if xmax > re_size[1] - 1:
            xmax = re_size[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > re_size[0] - 1:
            ymax = re_size[0] - 1

        width = xmax - xmin
        step = 16

        anchor_count = int(math.ceil(width / float(step)))
        for i in range(anchor_count):
            anchor_xmin = i * step + xmin
            anchor_xmax = anchor_xmin + step - 1
            if anchor_xmax >= re_size[1]:
                continue
            # plus one to get 1 based pascal voc data format.
            bboxes = np.vstack((bboxes, np.array([int(anchor_xmin)+1, ymin+1, int(anchor_xmax)+1, ymax+1])))

    write_xml(os.path.join(out_xml_path, imgbasefn + '.jpg'), re_size[0], re_size[1], bboxes)
    txt.write('%s\n' % imgbasefn)

def build_voc_dirs(outdir):
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    mkdir(outdir)
    mkdir(os.path.join(outdir, 'Annotations'))
    mkdir(os.path.join(outdir, 'ImageSets'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Layout'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Main'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Segmentation'))
    mkdir(os.path.join(outdir, 'JPEGImages'))
    return os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages'), os.path.join(outdir, 'ImageSets',
                                                                                                 'Main')

if __name__ == '__main__':
    in_path = r'/home/kk/data/mp'
    
    xmllist = [os.path.join(in_path, _) for _ in os.listdir(in_path) if _.endswith('.xml')]

    out_path = r'/home/kk/data/mpout'

    out_xml_path, out_img_path, out_main_path = build_voc_dirs(out_path)

    maintxt = open(os.path.join(out_main_path, 'train.txt'), 'w')
    for fn in xmllist:
        splitXml(fn, out_img_path, out_xml_path, maintxt)

    maintxt.close()
