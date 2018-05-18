#!/usr/bin/env python3
import cv2
import random
import numpy as np


def color_mask_graph (mask, dilate=100, num_C = 20, min_C = 1):
    '''
    Given a mask of object IDs, color the objects with color ID within [min_C, num_C), such that
    objects with a distance of dilate in the mask do not share the same color.
    Increase num_C when necessary.

    Returns dict  object ID -> color ID.
    '''

    # build adjacency graph
    graph = {}
    
    kernel = np.ones((dilate, dilate), np.uint8)
    for v in np.unique(mask):   # for each object
        if v == 0:  # background
            continue
        # neighborhood of the object
        nhood = np.copy(mask)
        nhood[cv2.dilate((mask == v).astype(np.uint8), kernel, iterations=1) == 0] = 0
        adj = []
        for v2 in np.unique(nhood):
            if v2 == 0 or v2 == v:
                continue
            adj.append(v2)
            pass
        graph[v] = adj
        pass

    colors = {}
    for v, adj in graph.items():
        used = [colors[v2] for v2 in adj if v2 in colors]
        avail = set(range(num_C)) - set(used)
        if len(avail) == 0:
            assert (min_C + num_C) < 256, 'color values are not enough'
            c = num_C
            num_C += 1
        else:
            c = random.sample(avail, 1)[0]
            pass
        colors[v] = c
        pass
    return {v: c+min_C for v, c in colors.items()}


def mask_to_annotation (id_mask, label_map, dilate=200, blur_K=5, blur_sigma=1, contour_th=0.5):
    # id_mask is of type np.int32, pixel value is object ID
    # labels is dict object ID -> label

    polys = []
    for oid, label in label_map.items():
        # for each object
        one = (mask == oid)
        # fill holes
        one = scipy.ndimage.morphology.binary_fill_holes(one)
        one = cv2.GaussianBlur(one.astype(np.float32), (blur_K,blur_K), blur_sigma)
        contours = measure.find_contours(one, contour_th)
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
    return {'shapes': polys}


def label_mask_to_annotation (mask, dilate=200, blur_K=5, blur_sigma=1, contour_th=0.5):
    # mask is of type np.int32, pixel value is object category

    H, W = mask.shape

    polys = []
    
    tag = 1 
    for label in np.unique(masks):
        if label == 0:
            continue
        # for each object
        one = (mask == label)
        # fill holes
        one = scipy.ndimage.morphology.binary_fill_holes(one)
        one = cv2.GaussianBlur(one.astype(np.float32), (blur_K,blur_K), blur_sigma)
        contours = measure.find_contours(one, contour_th)
        # use only the biggest contour
        # this is a compromize
        for contour in contours:
            points = []
            for y, x in contour:
                points.append({'x': 1.0 * x / W,
                              'y': 1.0 * y / H}) 
                pass
            polys.append({'type':'polygon', 'tag': tag, 'label': label, 'geometry':{'points': points}})
            tag += 1
        pass
    return {'shapes': polys}


