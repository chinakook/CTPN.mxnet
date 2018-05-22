import cv2

from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt

img = cv2.imread('/mnt/6B133E147DED759E/VOCdevkit/VOC2007/JPEGImages/img_1777.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

anno = ET.parse('/mnt/6B133E147DED759E/VOCdevkit/VOC2007/Annotations/img_1777.xml')

obj_node=anno.getiterator("object")

rects = []

for obj in obj_node:
    bndbox = obj.find('bndbox')
    xmin = bndbox.find('xmin')
    ymin = bndbox.find('ymin')
    xmax = bndbox.find('xmax')
    ymax = bndbox.find('ymax')
    rects.append(((int(xmin.text), int(ymin.text)), (int(xmax.text), int(ymax.text))))

for r in rects:
    cv2.rectangle(img, r[0], r[1], (0,255,0),1)

plt.imshow(img)
plt.show()