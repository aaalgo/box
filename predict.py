#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build/lib.linux-x86_64-3.5'))
import time
from tqdm import tqdm
import numpy as np
import cv2
from skimage import measure
import imageio
import subprocess as sp
# RESNET: import these for slim version of resnet
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import picpac
import cpp

class Model:
    def __init__ (self, X, anchor_th, nms_max, nms_th, is_training, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        self.boxes, self.masks = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X,
                               'anchor_th:0': anchor_th,
                               'nms_max:0': nms_max,
                               'nms_th:0': nms_th,
                               'is_training:0': is_training},
                    return_elements=['boxes:0', 'masks:0'])
        #self.prob = tf.squeeze(tf.slice(tf.nn.softmax(self.logits), [0,0,0,1], [-1,-1,-1,1]), 3)
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('input', None, '')
flags.DEFINE_string('output', 'output.gif', '')
flags.DEFINE_float('anchor_th', 0.5, '')
flags.DEFINE_integer('nms_max', 200, '')
flags.DEFINE_float('nms_th', 0.2, '')
flags.DEFINE_integer('stride', 16, '')
flags.DEFINE_string('shape', 'Circle', '')
flags.DEFINE_integer('channels', 3, '')

tableau20 = [(180, 119, 31), (232, 199, 174), (14, 127, 255), (120, 187, 255),
			 (44, 160, 44), (138, 223, 152), (40, 39, 214), (150, 152, 255),
			 (189, 103, 148), (213, 176, 197), (75, 86, 140), (148, 156, 196),
			 (194, 119, 227), (210, 182, 247), (127, 127, 127), (199, 199, 199),
			 (34, 189, 188), (141, 219, 219), (207, 190, 23), (229, 218, 158)]

def save_prediction_image (path, image, boxes, masks):
    images = []
    image = image.astype(np.uint8)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(rgb)
    #vis = np.zeros_like(rgb, dtype=np.uint8)
    vis = np.copy(rgb).astype(np.float32)

    boxes = np.round(boxes).astype(np.int32)

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0))
        patch = vis[y1:(y2+1), x1:(x2+1), :]

        mask = cv2.resize(masks[i], (x2-x1+1, y2-y1+1))
        view = vis[y1:(y2+1), x1:(x2+1)]
        b, g, r = tableau20[i % len(tableau20)]
        patch[:, :, 0][mask > 0.5] = 0
        patch[:, :, 1][mask > 0.5] = 0
        patch[:, :, 2][mask > 0.5] = 0
        patch[:, :, 0] += b * mask
        patch[:, :, 1] += g * mask
        patch[:, :, 2] += r * mask
    images.append(np.clip(vis, 0, 255).astype(np.uint8))
    imageio.mimsave(path + '.gif', images, duration = 1)
    sp.check_call('gifsicle --colors 256 -O3 < %s.gif > %s; rm %s.gif' % (path, path, path), shell=True)
    pass

def main (_):
    assert os.path.exists(FLAGS.input)
    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    is_training = tf.placeholder(tf.bool, name="is_training")
    anchor_th = tf.constant(FLAGS.anchor_th, tf.float32)
    nms_max = tf.constant(FLAGS.nms_max, tf.int32)
    nms_th = tf.constant(FLAGS.nms_th, tf.float32)
    model = Model(X, anchor_th, nms_max, nms_th, is_training, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model.loader(sess)
        image = cv2.imread(FLAGS.input, cv2.IMREAD_COLOR)
        H, W = image.shape[:2]
        H = H // FLAGS.stride * FLAGS.stride
        W = W // FLAGS.stride * FLAGS.stride
        image = image[:H, :W, :]
        batch = np.expand_dims(image, axis=0).astype(dtype=np.float32)
        boxes, masks = sess.run([model.boxes, model.masks], feed_dict={X: batch, is_training: False})
        print(boxes)
        save_prediction_image(FLAGS.output, image, boxes, masks)
    pass

if __name__ == '__main__':
    tf.app.run()

