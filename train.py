#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models/research/slim'))
sys.path.insert(0, 'build/lib.linux-x86_64-3.5')
import time
import datetime
import logging
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory, resnet_utils 
import picpac
import cpp
from gallery import Gallery

def patch_arg_scopes ():
    def resnet_arg_scope (weight_decay=0.0001):
        print_red("Patching resnet_v2 arg_scope when training from scratch")
        return resnet_utils.resnet_arg_scope(weight_decay=weight_decay,
                    batch_norm_decay=0.9,
                    batch_norm_epsilon=5e-4,
                    batch_norm_scale=False)
    nets_factory.arg_scopes_map['resnet_v1_50'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v1_101'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v1_152'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v1_200'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_50'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_101'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_152'] = resnet_arg_scope
    nets_factory.arg_scopes_map['resnet_v2_200'] = resnet_arg_scope
    pass

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor


augments = None
#from . config import *
#if os.path.exists('config.py'):
def print_red (txt):
    print('\033[91m' + txt + '\033[0m')

def print_green (txt):
    print('\033[92m' + txt + '\033[0m')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('db', None, 'training db')
flags.DEFINE_string('val_db', None, 'validation db')
flags.DEFINE_integer('classes', 2, 'number of classes')
flags.DEFINE_string('mixin', None, 'mix-in training db')
flags.DEFINE_integer('channels', 3, 'image channels')
flags.DEFINE_boolean('cache', True, '')

flags.DEFINE_integer('size', None, '') 
flags.DEFINE_integer('batch', 1, 'Batch size.  ')
flags.DEFINE_integer('shift', 0, '')
flags.DEFINE_integer('backbone_stride', 16, '')
flags.DEFINE_integer('anchor_logit_filters', 32, '')
flags.DEFINE_integer('anchor_params_filters', 64, '')
flags.DEFINE_integer('anchor_stride', 4, '')
flags.DEFINE_integer('mask_filters', 32, '')
flags.DEFINE_integer('mask_stride', 2, '')
flags.DEFINE_integer('mask_size', 128, '')
flags.DEFINE_float('anchor_th', 0.5, '')
flags.DEFINE_float('nms_th', 0.5, '')
flags.DEFINE_float('match_th', 0.5, '')
flags.DEFINE_integer('max_masks', 128, '')

flags.DEFINE_string('backbone', 'resnet_v2_50', 'architecture')
flags.DEFINE_string('model', None, 'model directory')
flags.DEFINE_string('resume', None, 'resume training from this model')
flags.DEFINE_string('finetune', None, '')
flags.DEFINE_integer('max_to_keep', 100, '')

# optimizer settings
flags.DEFINE_float('lr', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.95, '')
flags.DEFINE_float('decay_steps', 500, '')
flags.DEFINE_float('weight_decay', 0.00004, '')
#
flags.DEFINE_integer('epoch_steps', None, '')
flags.DEFINE_integer('max_epochs', 20000, '')
flags.DEFINE_integer('ckpt_epochs', 10, '')
flags.DEFINE_integer('val_epochs', 10, '')
flags.DEFINE_boolean('adam', False, '')

flags.DEFINE_float('pl_weight', 1.0/50, '')
flags.DEFINE_float('re_weight', 0.1, '')

COLORSPACE = 'BGR'
PIXEL_MEANS = tf.constant([[[[127.0, 127.0, 127.0]]]])
VGG_PIXEL_MEANS = tf.constant([[[[103.94, 116.78, 123.68]]]])

PRIORS = [1]    # placeholder

class Inputs:
    def __init__ (self):
        self.X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
        self.anchor_th = tf.placeholder(tf.float32, shape=(), name="anchor_th")
        self.nms_max = tf.placeholder(tf.int32, shape=(), name="nms_max")
        self.nms_th = tf.placeholder(tf.float32, shape=(), name="nms_th")
        # gt_xxx groundtruth
        self.gt_masks = tf.placeholder(tf.float32, shape=(None, None, None, 1))
        self.gt_anchors = tf.placeholder(tf.int32, shape=(None, None, None, len(PRIORS)))
        self.gt_anchors_weight = tf.placeholder(tf.float32, shape=(None, None, None, len(PRIORS)))
        self.gt_params = tf.placeholder(tf.float32, shape=(None, None, None, len(PRIORS) * 4))
        self.gt_params_weight = tf.placeholder(tf.float32, shape=(None, None, None, len(PRIORS)))
        self.gt_boxes = tf.placeholder(tf.float32, shape=(None, None))
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        pass

    # create feed_dict from a picpac sample
    def feed_dict (self, sample, is_training):
        _, images, gt_masks_, gt_anchors_, gt_anchors_weight_, gt_params_, gt_params_weight_, gt_boxes_ = sample  # unpack picpac sample
        return {self.X: images,
                self.anchor_th: FLAGS.anchor_th,
                self.nms_max: 1,
                self.nms_th: FLAGS.nms_th,
                self.gt_masks: gt_masks_,
                self.gt_anchors: gt_anchors_,
                self.gt_anchors_weight: gt_anchors_weight_,
                self.gt_params: gt_params_,
                self.gt_params_weight: gt_params_weight_,
                self.gt_boxes: gt_boxes_,
                self.is_training: is_training}
        pass


def anchors2boxes (shape, anchor_params):
    # anchor parameters are: dx, dy, w, h
    B = shape[0]
    H = shape[1]
    W = shape[2]
    box_ind = tf_repeat(tf.range(B), [H * W * len(PRIORS)])
    if True:    # generate array of box centers
        x0 = tf.cast(tf.range(W) * FLAGS.anchor_stride, tf.float32)
        y0 = tf.cast(tf.range(H) * FLAGS.anchor_stride, tf.float32)
        x0, y0 = tf.meshgrid(x0, y0)
        x0 = tf.reshape(x0, (-1,))
        y0 = tf.reshape(y0, (-1,))
        x0 = tf.tile(tf_repeat(x0, [len(PRIORS)]), [B])
        y0 = tf.tile(tf_repeat(y0, [len(PRIORS)]), [B])
    dx, dy, lw, lh = [tf.squeeze(x, axis=1) for x in tf.split(anchor_params, [1,1,1,1], 1)]

    W = tf.cast(W * FLAGS.anchor_stride, tf.float32)
    H = tf.cast(H * FLAGS.anchor_stride, tf.float32)

    max_X = W-1
    max_Y = H-1

    w = tf.clip_by_value(tf.exp(lw)-1, 0, W)
    h = tf.clip_by_value(tf.exp(lh)-1, 0, H)

    x1 = x0 + dx - w/2
    y1 = y0 + dy - h/2
    x2 = x1 + w
    y2 = y1 + h
    x1 = tf.clip_by_value(x1, 0, max_X) 
    y1 = tf.clip_by_value(y1, 0, max_Y)
    x2 = tf.clip_by_value(x2, 0, max_X)
    y2 = tf.clip_by_value(y2, 0, max_Y)

    boxes = tf.stack([x1, y1, x2, y2], axis=1)
    return boxes, box_ind

def normalize_boxes (shape, boxes):
    max_X = tf.cast(shape[2]-1, tf.float32)
    max_Y = tf.cast(shape[1]-1, tf.float32)
    x1,y1,x2,y2 = [tf.squeeze(x, axis=1) for x in tf.split(boxes, [1,1,1,1], 1)]
    x1 = x1 / max_X
    y1 = y1 / max_Y
    x2 = x2 / max_X
    y2 = y2 / max_Y
    return tf.stack([y1, x1, y2, x2], axis=1)

def shift_boxes (boxes, box_ind):
    assert FLAGS.batch == 1
    return boxes

def xxx_print (array):
    print(array)
    return np.zeros([1], dtype=np.float32)

def mask_net (X, mask_ft, boxes, box_ind):
    nboxes = normalize_boxes(tf.shape(X), boxes)
    net = tf.image.crop_and_resize(mask_ft, nboxes, box_ind, [FLAGS.mask_size, FLAGS.mask_size])

    with slim.arg_scope([slim.conv2d], normalizer_fn=None):
        net = slim.conv2d(net, 32, 3, 1, scope='masknet1', reuse=tf.AUTO_REUSE)
        net = slim.conv2d(net, 32, 3, 1, scope='masknet2', reuse=tf.AUTO_REUSE)
        net = slim.conv2d(net, 2, 3, 1, activation_fn=None, scope='masknet3', reuse=tf.AUTO_REUSE)
    return net

def create_model (inputs, backbone_fn):
    #box_ft, mask_ft, gt_masks, gt_anchors, gt_anchors_weight, gt_params, gt_params_weight, gt_boxes, config):
    # ft:           B * H' * W' * 3     input feature, H' W' is feature map size
    # gt_counts:    B                   number of boxes in each sample of the batch
    # gt_boxes:     ? * 4               boxes
    bb, _ = backbone_fn(inputs.X-PIXEL_MEANS, global_pool=False, output_stride=FLAGS.backbone_stride)
    #bb2, _ = backbone_fn(inputs.X-PIXEL_MEANS, global_pool=False, output_stride=FLAGS.backbone_stride, scope='bb2')

    gt_matcher = cpp.GTMatcher(FLAGS.match_th, FLAGS.max_masks)
    mask_extractor = cpp.MaskExtractor(FLAGS.mask_size, FLAGS.mask_size)

    end_points = {}

    with tf.variable_scope('boxnet'):

        assert FLAGS.backbone_stride % FLAGS.anchor_stride == 0
        ss = FLAGS.backbone_stride // FLAGS.anchor_stride
        # generate anchor feature
        anchor_logits_ft = slim.conv2d_transpose(bb, FLAGS.anchor_logit_filters, ss*2, ss)
        anchor_params_ft = slim.conv2d_transpose(bb, FLAGS.anchor_params_filters, ss*2, ss)

        assert FLAGS.backbone_stride % FLAGS.mask_stride == 0
        ss = FLAGS.backbone_stride // FLAGS.mask_stride

        mask_ft = slim.conv2d_transpose(bb, FLAGS.mask_filters, ss*2, ss)

        anchor_logits = slim.conv2d(anchor_logits_ft, 2 * len(PRIORS), 3, 1, activation_fn=None) 
        anchor_logits2 = tf.reshape(anchor_logits, (-1, 2))   # ? * 2
        # anchor probabilities
        anchor_prob = tf.squeeze(tf.slice(tf.nn.softmax(anchor_logits2), [0, 1], [-1, 1]), 1)

        gt_anchors = tf.reshape(inputs.gt_anchors, (-1, ))
        gt_anchors_weight = tf.reshape(inputs.gt_anchors_weight, (-1,))

        # anchor cross-entropy
        axe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=anchor_logits2, labels=gt_anchors)
        axe = axe * gt_anchors_weight
        axe = tf.reduce_sum(axe) / (tf.reduce_sum(gt_anchors_weight) + 1)

        params = slim.conv2d(anchor_params_ft, 4 * len(PRIORS), 3, 1, activation_fn=None)
        params = tf.reshape(params, (-1, 4))     # ? * 4
        gt_params = tf.reshape(inputs.gt_params, (-1, 4))
        gt_params_weight = tf.reshape(inputs.gt_params_weight, (-1,))
        # params loss
        if True:
            dxy, wh = tf.split(params, [2,2], 1)
            dxy_gt, wh_gt = tf.split(gt_params, [2,2], 1)

            #wh = tf.log(tf.nn.relu(wh) + 1)
            wh_gt = tf.log(wh_gt + 1)

            pl = tf.losses.huber_loss(dxy, dxy_gt, reduction=tf.losses.Reduction.NONE) + \
                 tf.losses.huber_loss(wh, wh_gt, reduction=tf.losses.Reduction.NONE)
            pl = tf.reduce_sum(pl, axis=1)

        pl = tf.reduce_sum(pl * gt_params_weight) / (tf.reduce_sum(gt_params_weight) + 1)

        # generate boxes from anchor params
        boxes, box_ind = anchors2boxes(tf.shape(anchor_logits_ft), params)
        boxes_pre = boxes

        sel = tf.greater_equal(anchor_prob, inputs.anchor_th)
        # sel is a boolean mask

        # select only boxes with prob > th for nms
        anchor_prob = tf.boolean_mask(anchor_prob, sel)
        boxes = tf.boolean_mask(boxes, sel)
        box_ind = tf.boolean_mask(box_ind, sel)

        sel = tf.image.non_max_suppression(shift_boxes(boxes, box_ind), anchor_prob, 100000, iou_threshold=inputs.nms_th)
        # sel is a list of indices

        if True:    # prediction head, not used in training
            psel = tf.slice(sel, [0], [tf.minimum(inputs.nms_max, tf.shape(sel)[0])])
            boxes_predicted = tf.gather(boxes, psel)
            box_ind_predicted = tf.gather(box_ind, psel)

            mlogits = mask_net(inputs.X, mask_ft, boxes_predicted, box_ind_predicted)
            masks_predicted = tf.squeeze(tf.slice(tf.nn.softmax(mlogits), [0, 0, 0, 1], [-1, -1, -1, 1]), 3)
            pass

        anchor_prob = None  # discard
        boxes = tf.gather(boxes, sel)
        box_ind = tf.gather(box_ind, sel)

        hit, index, gt_index = tf.py_func(gt_matcher.apply, [boxes, box_ind, inputs.gt_boxes], [tf.float32, tf.int32, tf.int32])


        # % boxes found
        precision = hit / tf.cast(tf.shape(boxes)[0] + 1, tf.float32);
        recall = hit / tf.cast(tf.shape(inputs.gt_boxes)[0] + 1, tf.float32);

        boxes = tf.gather(boxes, index)
        box_ind = tf.gather(box_ind, index)
        gt_boxes = tf.gather(inputs.gt_boxes, gt_index)

        # normalize boxes to [0-1]
        nboxes = normalize_boxes(tf.shape(inputs.X), boxes)
        mlogits = mask_net(inputs.X, mask_ft, boxes, box_ind)

        gt_masks, = tf.py_func(mask_extractor.apply, [inputs.gt_masks, gt_boxes, boxes], [tf.float32])
        #gt_masks, = tf.py_func(mask_extractor.apply, [inputs.gt_masks, gt_boxes, tf.slice(gt_boxes, [0, 3], [-1, 4])], [tf.float32])
        end_points['gt_boxes'] = gt_boxes
        end_points['boxes'] = boxes
        gt_masks = tf.cast(tf.round(gt_masks), tf.int32)
        end_points['gt_masks'] = gt_masks
        # mask cross entropy
        mxe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mlogits, labels=gt_masks)
        mxe = tf.reshape(mxe, (-1, ))
        mxe = tf.reduce_sum(mxe) / tf.cast(tf.shape(mxe)[0] + 1, tf.float32)

    #tf.identity(logits, name='logits')
    #tf.identity(params, name='params')
    #tf.identity(boxes_pre, name='boxes_pre')
    tf.identity(boxes_predicted, name='boxes')
    tf.identity(masks_predicted, name='masks')
    #tf.identity(mlogits, name='mlogits')
    axe = tf.identity(axe, name='ax') # cross-entropy
    mxe = tf.identity(mxe, name='mx') # cross-entropy
    pl = tf.identity(pl * FLAGS.pl_weight, name='pl') # params-loss
    reg = tf.identity(tf.reduce_sum(tf.losses.get_regularization_losses()) * FLAGS.re_weight, name='re')
    precision = tf.identity(precision, name='p')
    recall = tf.identity(recall, name='r')

    loss = tf.identity(axe + mxe + pl + reg, name='lo')

    return loss, [axe, mxe, pl, reg, precision, recall], end_points

def setup_finetune (ckpt, exclusions):
    print("Finetuning %s" % ckpt)
    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                print("Excluding %s" % var.op.name)
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)

    variables_to_train = []
    for scope in exclusions:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    print("Training %d out of %d variables" % (len(variables_to_train), len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))))
    if len(variables_to_train) < 10:
        for var in variables_to_train:
            print("    %s" % var.op.name)

    return slim.assign_from_checkpoint_fn(
            ckpt, variables_to_restore,
            ignore_missing_vars=False), variables_to_train


def create_picpac_stream (db_path, is_training):
    assert os.path.exists(db_path)
    augments = []
    if is_training:
        augments = [
                  #{"type": "augment.flip", "horizontal": True, "vertical": False},
                  {"type": "augment.rotate", "min":-10, "max":10},
                  {"type": "augment.scale", "min":0.9, "max":1.1},
                  {"type": "augment.add", "range":20},
                ]
    else:
        augments = []

    picpac_config = {"db": db_path,
              "loop": is_training,
              "shuffle": is_training,
              "reshuffle": is_training,
              "annotate": True,
              "channels": FLAGS.channels,
              "stratify": is_training,
              "dtype": "float32",
              "batch": FLAGS.batch,
              "colorspace": COLORSPACE,
              "cache": FLAGS.cache,
              "transforms": augments + [
                  {"type": "clip", "round": FLAGS.backbone_stride},
                  {"type": "anchors.dense.box", 'downsize': FLAGS.anchor_stride},
                  {"type": "box_feature"},
                  {"type": "rasterize", "use_tag": True, "dtype": "float32"}
                  ]
             }
    if is_training and not FLAGS.mixin is None:
        #print("mixin support is incomplete in new picpac.")
        assert os.path.exists(FLAGS.mixin)
        picpac_config['mixin'] = FLAGS.mixin
        picpac_config['mixin_group_reset'] = 0
        picpac_config['mixin_group_delta'] = 1
    return picpac.ImageStream(picpac_config)

def main (_):
    global PIXEL_MEANS

    logging.basicConfig(filename='train-%s-%s.log' % (FLAGS.backbone, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')),level=logging.DEBUG, format='%(asctime)s %(message)s')

    if FLAGS.model:
        try:
            os.makedirs(FLAGS.model)
        except:
            pass

    if FLAGS.finetune:
        print_red("finetune, using RGB with vgg pixel means")
        COLORSPACE = 'RGB'
        PIXEL_MEANS = VGG_PIXEL_MEANS
        if FLAGS.channels == 1:
            print_red("finetune requires us turning channels from 1 to 3")
        FLAGS.channels = 3

    inputs = Inputs()

    if not FLAGS.finetune:
        patch_arg_scopes()

    backbone_fn = nets_factory.get_network_fn(FLAGS.backbone, num_classes=None,
                weight_decay=FLAGS.weight_decay, is_training=inputs.is_training)

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d], padding='SAME'), \
         slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=slim.l2_regularizer(2.5e-4), normalizer_fn=slim.batch_norm, normalizer_params={'decay': 0.9, 'epsilon': 5e-4, 'scale': False, 'is_training':inputs.is_training}), \
         slim.arg_scope([slim.batch_norm], is_training=inputs.is_training):
        loss, metrics, end_points = create_model(inputs, backbone_fn)

    metric_names = [x.name[:-2] for x in metrics]

    def format_metrics (avg):
        return ' '.join(['%s=%.3f' % (a, b) for a, b in zip(metric_names, list(avg))])

    init_finetune, variables_to_train = None, None
    if FLAGS.finetune:
        print_red("finetune, using RGB with vgg pixel means")
        init_finetune, variables_to_train = setup_finetune(FLAGS.finetune, [FLAGS.net + '/logits'])

    global_step = tf.train.create_global_step()
    LR = tf.train.exponential_decay(FLAGS.lr, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    if FLAGS.adam:
        print("Using Adam optimizer, reducing LR by 100x")
        optimizer = tf.train.AdamOptimizer(LR/100)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=0.9)

    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step, variables_to_train=variables_to_train)
    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    stream = create_picpac_stream(FLAGS.db, True)
    # load validation db
    val_stream = None
    if FLAGS.val_db:
        val_stream = create_picpac_stream(FLAGS.val_db, False)

    epoch_steps = FLAGS.epoch_steps
    if epoch_steps is None:
        epoch_steps = (stream.size() + FLAGS.batch-1) // FLAGS.batch
    best = 0

    ss_config = tf.ConfigProto()
    ss_config.gpu_options.allow_growth=True
    with tf.Session(config=ss_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if init_finetune:
            init_finetune(sess)
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)

        global_start_time = time.time()
        epoch = 0
        step = 0
        while epoch < FLAGS.max_epochs:
            start_time = time.time()
            cnt, metrics_sum = 0, np.array([0] * len(metrics), dtype=np.float32)
            progress = tqdm(range(epoch_steps), leave=False)
            for _ in progress:
                sample = stream.next()
                mm, _, ccc = sess.run([metrics, train_op, end_points['gt_masks']], feed_dict=inputs.feed_dict(sample, True))
                bs = sample[1].shape[0]
                metrics_sum += np.array(mm) * bs
                cnt += bs
                metrics_txt = format_metrics(metrics_sum/cnt)
                progress.set_description(metrics_txt)
                step += 1

                '''
                if ccc.shape[0] > 5:
                    gal = Gallery('ccc')
                    for i in range(ccc.shape[0]):
                        cv2.imwrite(gal.next(), ccc[i]*255)
                    gal.flush()
                    sys.exit(0)
                '''
                pass
            stop = time.time()
            msg = 'train e=%d s=%d ' % (epoch, step)
            msg += metrics_txt
            msg += ' w=%.3f t=%.3f ' % (stop - global_start_time, stop - start_time)
            print_green(msg)
            logging.info(msg)

            epoch += 1

            if (epoch % FLAGS.val_epochs == 0) and val_stream:
                lr = sess.run(LR)
                # evaluation
                Ys, Ps = [], []
                cnt, metrics_sum = 0, np.array([0] * len(metrics), dtype=np.float32)
                val_stream.reset()
                progress = tqdm(val_stream, leave=False)
                for sample in progress:
                    p, mm = sess.run([probs, metrics], feed_dict=inputs.feed_dict(sample, False))
                    metrics_sum += np.array(mm) * images.shape[0]
                    cnt += images.shape[0]
                    Ys.extend(list(meta.labels))
                    Ps.extend(list(p))
                    metrics_txt = format_metrics(metrics_sum/cnt)
                    progress.set_description(metrics_txt)
                    pass
                assert cnt == val_stream.size()
                avg = metrics_sum / cnt
                if avg[0] > best:
                    best = avg[0]
                msg = 'valid epoch=%d step=%d ' % (epoch-1, step)
                msg += metrics_txt
                msg += ' lr=%.4f best=%.3f' % (lr, best)
                print_red(msg)
                logging.info(msg)
                #log.write('%d\t%s\t%.4f\n' % (epoch, '\t'.join(['%.4f' % x for x in avg]), best))
            # model saving
            if (epoch % FLAGS.ckpt_epochs == 0) and FLAGS.model:
                ckpt_path = '%s/%d' % (FLAGS.model, epoch)
                saver.save(sess, ckpt_path)
                print('saved to %s.' % ckpt_path)
            pass
        pass
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

