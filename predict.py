#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, 'build/lib.linux-x86_64-3.5')
import time
from tqdm import tqdm
import numpy as np
import cv2
from skimage import measure
# RESNET: import these for slim version of resnet
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import cpp

class Model:
    def __init__ (self, X, is_training, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        self.logits, self.params = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X, 'is_training:0': is_training},
                    return_elements=['logits:0', 'params:0'])
        self.prob = tf.squeeze(tf.slice(tf.nn.softmax(self.logits), [0,0,0,1], [-1,-1,-1,1]), 3)
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('input', None, '')
flags.DEFINE_float('cth', 0.5, '')
flags.DEFINE_float('th', 0.5, '')
flags.DEFINE_integer('stride', 16, '')
flags.DEFINE_string('shape', 'Circle', '')


def save_prediction_image (path, image, prob, params):
    # image: original input image
    # prob: probability
    H, W = image.shape[:2]
    Hm, Wm = prob.shape[:2]
    assert H % Hm == 0
    assert W % Wm == 0
    assert H // Hm == W // Wm
    prop = getattr(cpp, FLAGS.shape + 'Proposal')(H//Hm, FLAGS.cth, FLAGS.th)
    vis = np.copy(image).astype(np.float32)
    x = prop.apply(prob, params, vis)

    cv2.imwrite(path, vis)
    print(x[:, 2] - x[:, 0])
    print(x[:, 3] - x[:, 1])
    pass

def main (_):
    assert os.path.exists(FLAGS.input)
    X = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="images")
    is_training = tf.placeholder(tf.bool, name="is_training")
    model = Model(X, is_training, FLAGS.model, 'xxx')
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
        prob, params = sess.run([model.prob, model.params], feed_dict={X: batch, is_training: False})
        save_prediction_image(FLAGS.input + '.prob.png', image, prob[0], params[0])
    pass

if __name__ == '__main__':
    tf.app.run()

