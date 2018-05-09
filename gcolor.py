#!/usr/bin/env python3
import cv2
import random
import numpy as np

def color_mask_graph (mask, dilate=20, num_C = 20, min_C = 1):
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
            c = num_C
            num_C += 1
        else:
            c = random.sample(avail, 1)[0]
            pass
        colors[v] = c
        pass
    return {v: c+min_C for v, c in colors.items()}

