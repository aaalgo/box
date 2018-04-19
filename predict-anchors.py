#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from tqdm import tqdm
import numpy as np
import cv2
from skimage import measure
# RESNET: import these for slim version of resnet
import tensorflow as tf
from tensorflow.python.framework import meta_graph

class Model:
    def __init__ (self, X, anchor_th, nms_max, nms_th, is_training, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        self.boxes, self.probs = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X,
                               'anchor_th:0': anchor_th,
                               'nms_max:0': nms_max,
                               'nms_th:0': nms_th,
                               'is_training:0': is_training},
                    return_elements=['boxes:0', 'probs:0'])
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('input', None, '')
flags.DEFINE_string('input_db', None, '')
flags.DEFINE_integer('stride', 16, '')

flags.DEFINE_float('anchor_th', 0.5, '')
flags.DEFINE_integer('nms_max', 200, '')
flags.DEFINE_float('nms_th', 0.2, '')
flags.DEFINE_float('max', None, 'max images from db')

def save_prediction_image (path, image, boxes, probs):
    boxes = np.round(boxes).astype(np.int32)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))

    cv2.imwrite(path, image)
    pass

def main (_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="images")
    is_training = tf.placeholder(tf.bool, name="is_training")
    anchor_th = tf.constant(FLAGS.anchor_th, tf.float32)
    nms_max = tf.constant(FLAGS.nms_max, tf.int32)
    nms_th = tf.constant(FLAGS.nms_th, tf.float32)
    model = Model(X, anchor_th, nms_max, nms_th, is_training, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model.loader(sess)
        if FLAGS.input:
            assert os.path.exists(FLAGS.input)
            image = cv2.imread(FLAGS.input, cv2.IMREAD_COLOR)
            batch = np.expand_dims(image, axis=0).astype(dtype=np.float32)
            boxes, probs = sess.run([model.boxes, model.probs], feed_dict={X: batch, is_training: False})
            save_prediction_image(FLAGS.input + '.prob.png', image, boxes, probs)
        if FLAGS.input_db:
            assert os.path.exists(FLAGS.input_db)
            import picpac
            from gallery import Gallery
            picpac_config = {"db": FLAGS.input_db,
                      "loop": False,
                      "shuffle": False,
                      "reshuffle": False,
                      "annotate": False,
                      "channels": 3,
                      "stratify": False,
                      "dtype": "float32",
                      "batch": 1,
                      "transforms": []
                     }
            stream = picpac.ImageStream(picpac_config)
            gal = Gallery(FLAGS.input_db + '.out')
            C = 0
            for _, images in stream:
                boxes, probs = sess.run([model.boxes, model.probs], feed_dict={X: images, is_training: False})
                save_prediction_image(gal.next(), images[0], boxes, probs)
                C += 1
                if FLAGS.max and C >= FLAGS.max:
                    break
                pass
            pass
            gal.flush()
    pass

if __name__ == '__main__':
    tf.app.run()

