#!/sr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.ghostnet import GhostNet
from lib.utils.timer import Timer
from tensorpack.tfutils.tower import TowerContext

#CLASSES = ('__background__', 'fake-rice', 'fake-huo_tui_mu_er_chao_dan', 'fake-qing_chao_si_ji_dou', #'fake-gong_bao_ji_ding', 
#		   'fake-fan_qie_chao_dan', 'fake-chao_xi_lan_hua', 'fake-hong_shao_rou', 'fake-suan_la_tu_dou_si', #'fake-xiao_long_bao',
#		   'fake-suan_la_bai_cai', 'fake-ji_pai_fan', 'fake-niu_pai', 'fake-zhu_pa_fan', 
#		   'fake-jin_zhen_gu_xia', 'bread-pineapple', 'bread-mediumhorn', 'bread-largehorn', 
#		   'bread-pattern', 'bread-sausage', 'bread-prismatic',
#		   'bread-hempflower', 'bread-caterpillar', 'bread-smallmeal', 'toast', 'donut', 'hamburger-opening')

pclslist = ['__background__']
path = r'D:\Faster-R-CNN\labels_13_BJOYYX.txt'
with open(path,'r') as frcls:
    for line in frcls:
        line = line.split()
        #print ('line is : ', line[0])
        pclslist.append(line[0])
    #self._classes = ('__background__',  # always index 0
    #                 'newcp-rectangle_small_plate')
classes_num = int(len(pclslist))
CLASSES = tuple(pclslist)

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


#def vis_detections(im, class_name, dets, thresh=0.5):
#    """Draw detected bounding boxes."""
#    inds = np.where(dets[:, -1] >= thresh)[0]
#    if len(inds) == 0:
#        return
#
#    im = im[:, :, (2, 1, 0)]
#    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(im, aspect='equal')
#    for i in inds:
#        bbox = dets[i, :4]
#        score = dets[i, -1]
#
#        ax.add_patch(
#            plt.Rectangle((bbox[0], bbox[1]),
#                          bbox[2] - bbox[0],
#                          bbox[3] - bbox[1], fill=False,
#                          edgecolor='red', linewidth=3.5)
#        )
#        ax.text(bbox[0], bbox[1] - 2,
#                '{:s} {:.3f}'.format(class_name, score),
#                bbox=dict(facecolor='blue', alpha=0.5),
#                fontsize=14, color='white')
#
#    ax.set_title(('{} detections with '
#                  'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                  thresh),
#                 fontsize=14)
#    plt.axis('off')
#    plt.tight_layout()
#    plt.draw()
    

    
def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im_file = os.path.join(path1, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.1
    thresh = CONF_THRESH
    
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH) 
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
    
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(cls, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    os.chdir(path2)
    plt.savefig(im_name)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res50 mobilenetv1]',
                        choices=NETS.keys(), default='mobilenetv1')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = r'D:\Faster-R-CNN\default\voc_2007_trainval\default_555_555\mobilenetv1_faster_rcnn_iter_100000.ckpt'
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'mobilenetv1':
        net = GhostNet(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    with TowerContext('', is_training=False):
        net.create_architecture(sess, "TEST", classes_num,
                                tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    #im_names = ['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg',
    #           '000005.jpg', '000006.jpg']
    
    path1 = input("请输入测试图片的路径:")
    im_names = os.listdir(path1)
    path2 = r'D:\Faster-R-CNN\output'
    if not os.path.exists(path2):
        os.makedirs(path2)
    
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
    #plt.show()
