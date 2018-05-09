#!/usr/bin/env python3
import os
import sys
sys.path.append('..')
import picpac
import simplejson as json
import cv2
import numpy as np
from skimage import measure
import scipy
from gcolor import *
from lyft import *

json.encoder.FLOAT_REPR = lambda f: ("%.3f" % f)
json.encoder.c_make_encoder = None

BASE = os.path.abspath(os.path.dirname(__file__))

assert len(LABEL_MAP) == 1

db = picpac.Writer('scratch/train.db', picpac.OVERWRITE)

CC = 0
with open(os.path.join(BASE, 'train.list'), 'r') as f:
    for line in f:
        stem = line.strip()

        image_path = os.path.join(BASE, 'data/Train/CameraRGB/%s.png' % stem)
        seg_path = os.path.join(BASE, 'data/Train/CameraSeg/%s.png' % stem)
        print(seg_path)
        assert os.path.exists(image_path)
        assert os.path.exists(seg_path)

        #image = cv2.imread(image_path)
        seg= cv2.imread(seg_path)
        #assert image.shape == seg.shape
        assert np.all(seg[:,:,:2] == 0)
        #print(seg.dtype, seg.shape)
        #print(np.unique(seg[:,:,2]))

        H, W = seg.shape[:2]
        # for each label we want

        min_tag = 1  # tag offset.  we need to make sure objects of different
                     # categories do not share the same tags

        polys = []   # polygon annotations
        for from_label, to_label in LABEL_MAP.items():

            # label objects of this category as 1, 2, ...
            mask, nobj = measure.label(seg[:, :, 2] == from_label, background=0, return_num=True)
            if nobj == 0:
                continue

            tags = color_mask_graph(mask, min_C = min_tag)
            assert len(tags) == nobj
            assert np.max(mask) == nobj
            min_tag = max(tags.values()) + 1    # set for next iteration
            assert min_tag <= 256               # so all tags can be represted by uint8

            for v in range(1, nobj+1):   # for each object
                # for each object
                one = (mask == v)
                if IGNORE_SELF_CAR:
                    if np.sum(one[:SELF_CAR_H,:]) < 10:
                    # object too small          # TODO: is it good to ignore small objects?
                        continue
                else:
                    if np.sum(one) < 10:
                        continue
                # fill holes
                one = scipy.ndimage.morphology.binary_fill_holes(one)
                one = cv2.GaussianBlur(one.astype(np.float32), (5,5), 1)
                contours = measure.find_contours(one, 0.2)
                # use only the biggest contour
                # this is a compromize
                if len(contours) > 1:
                    print("Contours: ", len(contours))
                if len(contours) == 0:
                    assert False
                    #assert np.sum(mask == v) < 10  # no contour found, must be very small
                    #continue
                best = []
                for contour in contours:
                    if len(contour) > len(best):
                        best = contour
                points = []
                for y, x in best:
                    points.append({'x': 1.0 * x / W,
                                  'y': 1.0 * y / H}) 
                    pass
                polys.append({'type':'polygon', 'tag': tags[v], 'label': to_label, 'geometry':{'points': points}})
                pass
        print("SHAPES: %d" % len(polys))
        anno = {'shapes': polys}
        with open(image_path, 'rb') as f:
            image_buf = f.read()
        anno_buf = json.dumps(anno).encode('ascii')
        print(len(anno_buf))
        db.append(0, image_buf, anno_buf)
        CC += 1

