# --------------------------------------------------------
# Faster R-CNN GhostNet
# Written by YLF
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import namedtuple
import functools
import numpy as np

import tensorflow as tf

from tensorpack.models import (
    MaxPooling, GlobalAvgPooling, BatchNorm, Dropout, BNReLU, FullyConnected)
from tensorpack.tfutils import argscope
from tensorpack.tfutils.tower import TowerContext

from tensorpack.models.common import layer_register
from tensorpack.utils.argtools import shape2d

#from imagenet_utils import ImageNetModel
import lib.nets.utils as utils
from lib.nets.myconv2d import MyConv2D as Conv2D
from lib.nets.myconv2d import BNNoReLU, SELayer
from lib.nets.myconv2d import GhostModule as MyConv

from lib.nets.network import Network
from lib.config import config as cfg

kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)

slim = tf.contrib.slim



@layer_register(log_shape=True)
def DepthConv(x, kernel_shape, padding='SAME', stride=1, data_format='NHWC',
              W_init=None, activation=tf.identity):
    in_shape = x.get_shape().as_list()
    if data_format=='NHWC':
        in_channel = in_shape[3]
        stride_shape = [1, stride, stride, 1]
    elif data_format=='NCHW':
        in_channel = in_shape[1]
        stride_shape = [1, 1, stride, stride]
    out_channel = in_channel
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = kernel_initializer
    kernel_shape = shape2d(kernel_shape) #[kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, stride_shape, padding=padding, data_format=data_format)
    return activation(conv, name='output')


Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'factor', 'se'])
Bottleneck = namedtuple('Bottleneck', ['kernel', 'stride', 'depth', 'factor', 'se'])

# _CONV_DEFS specifies the GhostNet body
_CONV_DEFS_0 = [
    Conv(kernel=[3, 3], stride=2, depth=16, factor=1, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=16, factor=1, se=0),

    Bottleneck(kernel=[3, 3], stride=2, depth=24, factor=48/16, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=24, factor=72/24, se=0),

    Bottleneck(kernel=[5, 5], stride=2, depth=40, factor=72/24, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=40, factor=120/40, se=1),

    Bottleneck(kernel=[3, 3], stride=2, depth=80, factor=240/40, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=200/80, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184/80, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184/80, se=0),

    Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=480/80, se=1),
    Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=672/112, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=672/112, se=1),

    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960/160, se=0),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960/160, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960/160, se=0),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960/160, se=1),

    Conv(kernel=[1, 1], stride=1, depth=960, factor=1, se=0),
    Conv(kernel=[1, 1], stride=1, depth=1280, factor=1, se=0)
]


def ghostnet_base(inputs,
                  final_endpoint=None,
                  min_depth=8,
                  depth_multiplier=1.0,
                  depth=1.0,
                  conv_defs=None,
                  output_stride=None,
                  dw_code=None,
                  ratio_code=None,
                  se=1,
                  scope=None,
                  starting_layer=0):
    def depth(d):
        d = max(int(d * depth_multiplier), min_depth)
        d = round(d / 4) * 4
        return d

    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS_0

    if dw_code is None or len(dw_code) < len(conv_defs):
        dw_code = [3] * len(conv_defs)
    print('dw_code', dw_code)

    if ratio_code is None or len(ratio_code) < len(conv_defs):
        ratio_code = [2] * len(conv_defs)
    print('ratio_code', ratio_code)

    se_code = [x.se for x in conv_defs]
    print('se_code', se_code)

    if final_endpoint is None:
        final_endpoint = 'Conv2d_%d' % (len(conv_defs) - 1)

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    with tf.variable_scope(scope, 'MobileNetV2', [inputs]):
        # with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1
            net = inputs
            in_depth = 3
            gi = 0
            for i, conv_def in enumerate(conv_defs):
                print('---')
                end_point_base = 'Conv2d_%d' % (i + starting_layer)
                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride

                # change last bottleneck
                if i + 2 == len(conv_defs):
                    end_point = end_point_base
                    net = Conv2D(end_point, net, depth(conv_def.depth), [1, 1], stride=1,
                                 data_format='NHWC', activation=BNReLU, use_bias=False)
                    #
                    # ksize = utils.ksize_for_squeezing(net, 1024)
                    # net = slim.avg_pool2d(net, ksize, padding='VALID',
                    #                       scope='AvgPool_7')
                    # end_points[end_point] = net

                # Normal conv2d.
                elif i + 1 == len(conv_defs):
                    end_point = end_point_base
                    net = Conv2D(end_point, net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride,
                                 data_format='NHWC', activation=BNReLU, use_bias=False)
                    end_points[end_point] = net

                elif isinstance(conv_def, Conv):
                    end_point = end_point_base
                    net = Conv2D(end_point, net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride,
                                 data_format='NHWC', activation=BNReLU, use_bias=False)
                    end_points[end_point] = net

                # Bottleneck block.
                elif isinstance(conv_def, Bottleneck):
                    # Stride > 1 or different depth: no residual part.
                    if layer_stride == 1 and in_depth == conv_def.depth:
                        res = net
                    else:
                        end_point = end_point_base + '_shortcut_dw'
                        res = DepthConv(end_point, net, conv_def.kernel, stride=layer_stride,
                                        data_format='NHWC', activation=BNNoReLU)
                        end_point = end_point_base + '_shortcut_1x1'
                        res = Conv2D(end_point, res, depth(conv_def.depth), [1, 1], strides=1, data_format='NHWC',
                                     activation=BNNoReLU, use_bias=False)

                    # Increase depth with 1x1 conv.
                    end_point = end_point_base + '_up_pointwise'
                    net = MyConv(end_point, net, depth(in_depth * conv_def.factor), [1, 1], dw_code[gi], ratio_code[gi],
                                 strides=1, data_format='NHWC', activation=BNReLU, use_bias=False)
                    end_points[end_point] = net

                    # Depthwise conv2d.
                    if layer_stride > 1:
                        end_point = end_point_base + '_depthwise'
                        net = DepthConv(end_point, net, conv_def.kernel, stride=layer_stride,
                                        data_format='NHWC', activation=BNNoReLU)
                        end_points[end_point] = net
                    # SE
                    if se_code[i] > 0 and se > 0:
                        end_point = end_point_base + '_se'
                        net = SELayer(end_point, net, depth(in_depth * conv_def.factor), 4)
                        end_points[end_point] = net

                    # Downscale 1x1 conv.
                    end_point = end_point_base + '_down_pointwise'
                    net = MyConv(end_point, net, depth(conv_def.depth), [1, 1], dw_code[gi], ratio_code[gi], strides=1,
                                 data_format='NHWC', activation=BNNoReLU if res is None else BNNoReLU, use_bias=False)
                    gi += 1

                    # Residual connection?
                    end_point = end_point_base + '_residual'
                    net = tf.add(res, net, name=end_point) if res is not None else net
                    end_points[end_point] = net

                # Unknown...
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
                in_depth = conv_def.depth
                # Final end point?
                if final_endpoint in end_points:
                    return end_points[final_endpoint], end_points

    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def ghostnet_arg_scope(is_training=True,
                       data_format='NHWC',
                       weight_decay=0.00004,
                       use_batch_norm=True,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       stddev=0.09,
                       regularize_depthwise=False):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': True,
        'scale': True,
        'data_format': data_format,
        'is_training': False,
        'trainable': False,

    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}

    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    if regularize_depthwise:
        depthwise_regularizer = weights_regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        trainable=is_training,
                        weights_initializer=weights_initializer,
                        activation_fn=tf.nn.relu6,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=normalizer_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer):
                    # Data format scope...
                    data_sc = utils.data_format_scope(data_format)
                    with slim.arg_scope(data_sc) as sc:
                        return sc

class GhostNet(Network):
    def __init__(self, batch_size=1, data_format='NHWC',
                 width=1.0, lr=0.001, weight_decay = 0.0005,
                 label_smoothing=0.0):
        Network.__init__(self, batch_size=batch_size)
        self.dropout_keep_prob = 0.8
        self.is_training = True
        self.min_depth = 8
        self.depth = 1.0
        self.conv_defs = None
        self.spatial_squeeze = True
        self.reuse = None
        self._scope = 'MobileNetV2'
        self.global_pool = False
        self.dw_code = None
        self.ratio_code = None
        self.se = 1
        self.data_format = data_format
        self.lr = lr
        self.depth_multiplier = width
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

    def build_network(self, sess, is_training=True):
        if cfg.FLAGS.initializer == "truncated":
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        net_conv = self._image


        # with tf.variable_scope(self._scope, self._scope) as scope:
        # sc1 = ghostnet_arg_scope(
        #     is_training=False,
        #     data_format=self.data_format,
        #     weight_decay=self.weight_decay,
        #     use_batch_norm=True,
        #     batch_norm_decay=0.9997,
        #     batch_norm_epsilon=0.001,
        #     regularize_depthwise=False)
        # with slim.arg_scope(sc1):
        with argscope(Conv2D,
                      kernel_initializer=initializer):
            with argscope([Conv2D, BatchNorm], data_format=self.data_format):


                # with slim.arg_scope([slim.batch_norm, slim.dropout],
                #                     is_training=is_training):
                net_conv, end_points = ghostnet_base(net_conv, scope=self._scope, dw_code=self.dw_code, ratio_code=self.ratio_code,
                                                se=1, min_depth=8, depth=1.0,
                                                depth_multiplier=1.0,
                                                conv_defs=_CONV_DEFS_0[:16])


        self._act_summaries.append(net_conv)
        self._layers['head'] = net_conv

        with tf.variable_scope(self._scope, self._scope):
            # build the anchors for the image
            self._anchor_component()
            rpn = slim.conv2d(net_conv, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                              scope="rpn_conv/3x3")
            self._act_summaries.append(rpn)
            rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                        weights_initializer=initializer,
                                        padding='VALID', activation_fn=None, scope='rpn_cls_score')
            # change it so that the score has 2 as its channel size
            rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
            rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
            rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
            rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                        weights_initializer=initializer,
                                        padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
            if is_training:
                rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
                # Try to have a determinestic order for the computing graph, for reproducibility
                with tf.control_dependencies([rpn_labels]):
                    rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
            else:
                if cfg.FLAGS.test_mode == 'nms':
                    rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                elif cfg.FLAGS.test_mode == 'top':
                    rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                else:
                    raise NotImplementedError


            pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
            #
            # sc2 = ghostnet_arg_scope(
            #     is_training=False,
            #     data_format=self.data_format,
            #     weight_decay=self.weight_decay,
            #     use_batch_norm=True,
            #     batch_norm_decay=0.9997,
            #     batch_norm_epsilon=0.001,
            #     regularize_depthwise=False)
            # with slim.arg_scope(sc2):
            with argscope(Conv2D,
                          kernel_initializer=initializer):
                with argscope([Conv2D, BatchNorm], data_format=self.data_format):
                #with tf.variable_scope(self._scope, self._scope) as scope:
                    # with slim.arg_scope([slim.batch_norm, slim.dropout],
                    #                     is_training=is_training):
                    fc7, end_points2 = ghostnet_base(pool5, scope=self._scope, dw_code=self.dw_code,
                                                         ratio_code=self.ratio_code,
                                                         se=1, min_depth=8, depth=1.0,
                                                         depth_multiplier=1.0,
                                                         conv_defs=_CONV_DEFS_0[17:])

            with tf.variable_scope(self._scope, self._scope):
                # average pooling done by reduce_mean
                fc7 = tf.reduce_mean(fc7, axis=[1, 2])
                cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer,
                                                 trainable=is_training, activation_fn=None, scope='cls_score')
                cls_prob = self._softmax_layer(cls_score, "cls_prob")
                bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                                 trainable=is_training, activation_fn=None,
                                                 scope='bbox_pred')
            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred
            self._predictions["rois"] = rois

            self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/Conv2d_0/W:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix MobileNet V1 layers..')
        with tf.variable_scope('Fix_MobileNet_V2') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR, and match the scale by (255.0 / 2.0)
                Conv2d_0_rgb = tf.get_variable("Conv2d_0_rgb",
                                               [3, 3, 3, 16],
                                               trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/Conv2d_0/W": Conv2d_0_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + "/Conv2d_0/W:0"],
                                   tf.reverse(Conv2d_0_rgb,[2])))


