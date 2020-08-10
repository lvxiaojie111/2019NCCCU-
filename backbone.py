#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : ljj1991
#   Created date: 2019-11-17 11:03:35
#   Description :
#backbone.py/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/backbone.py
#================================================================

import common
import tensorflow as tf
import cv2
import numpy as np
def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        #input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 3,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(1):#2
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+3))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(1):#8
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+6))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(1):#8
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                           trainable=trainable, name='conv43', downsample=True)
        #
        for i in range(1):#4
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+13))



        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable, name='conv0')
        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 64),
        #                                   trainable=trainable, name='conv1', downsample=True)

        # for i in range(1):
        #     input_data = common.residual_block(input_data, 64, 32, 64, trainable=trainable, name='residual%d' % (i + 0))
        #
        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 128),
        #                                   trainable=trainable, name='conv4', downsample=True)
        #
        # for i in range(1):  # 2
        #     input_data = common.residual_block(input_data, 128, 64, 128, trainable=trainable,
        #                                        name='residual%d' % (i + 1))
        #
        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
        #                                   trainable=trainable, name='conv9', downsample=True)
        #
        # for i in range(1):  # 8
        #     input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable,
        #                                        name='residual%d' % (i + 3))
        #
        # route_1 = input_data
        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
        #                                   trainable=trainable, name='conv26', downsample=True)
        #
        # for i in range(1):  # 8
        #     input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable,
        #                                        name='residual%d' % (i + 11))
        #
        # route_2 = input_data
        # input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
        #                                   trainable=trainable, name='conv43', downsample=True)
        #
        # for i in range(1):  # 4
        #     input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable,
        #                                        name='residual%d' % (i + 19))

        print("1235",input_data)
        return route_1, route_2, input_data






