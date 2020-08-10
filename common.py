#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 09:56:29
#   Description :
#
#================================================================

import tensorflow as tf


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.01))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)
        #if activate == True: conv = tf.nn.relu(conv)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(3, 3, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output

##########################################################################
########describle:      resnet50 module 11/14/2019
##########################################################################
def conv_op(x, name, n_out, training, useBN, kh=3, kw=3, dh=1, dw=1, padding="SAME", activation=tf.nn.relu,
            ):
    '''
    x:输入
    kh,kw:卷集核的大小
    n_out:输出的通道数
    dh,dw:strides大小
    name:op的名字

    '''
    #print(name,x)
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
        z = tf.nn.bias_add(conv, b)
        if useBN:
            z = tf.layers.batch_normalization(z, trainable=training)
        if activation:
            z = activation(z)
        return z


def max_pool_op(x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    return tf.nn.max_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def avg_pool_op(x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    return tf.nn.avg_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def fc_op(x, name, n_out, activation=tf.nn.relu):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[n_in, n_out],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))

        fc = tf.matmul(x, w) + b

        out = activation(fc)

    return fc, out


def res_block_layers(x, name, n_out_list, change_dimension=False, block_stride=1):
    if change_dimension:
        short_cut_conv = conv_op(x, name + "_ShortcutConv", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                                 dh=block_stride, dw=block_stride,
                                 padding="SAME", activation=None)
    else:
        short_cut_conv = x

    block_conv_1 = conv_op(x, name + "_lovalConv1", n_out_list[0], training=True, useBN=True, kh=1, kw=1,
                           dh=block_stride, dw=block_stride,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_2 = conv_op(block_conv_1, name + "_lovalConv2", n_out_list[0], training=True, useBN=True, kh=3, kw=3,
                           dh=1, dw=1,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_3 = conv_op(block_conv_2, name + "_lovalConv3", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                           dh=1, dw=1,
                           padding="SAME", activation=None)

    block_res = tf.add(short_cut_conv, block_conv_3)
    res = tf.nn.relu(block_res)
    return res

def bulid_resNet(x, num_class, training=True, usBN=True):
    conv1 = conv_op(x, "conv1", 64, training, usBN, 3, 3, 1, 1)
    pool1 = max_pool_op(conv1, "pool1", kh=3, kw=3)

    block1_1 = res_block_layers(pool1, "block1_1", [64, 256], True, 1)
    block1_2 = res_block_layers(block1_1, "block1_2", [64, 256], False, 1)
    block1_3 = res_block_layers(block1_2, "block1_3", [64, 256], False, 1)

    block2_1 = res_block_layers(block1_3, "block2_1", [128, 512], True, 2)
    block2_2 = res_block_layers(block2_1, "block2_2", [128, 512], False, 1)
    block2_3 = res_block_layers(block2_2, "block2_3", [128, 512], False, 1)
    block2_4 = res_block_layers(block2_3, "block2_4", [128, 512], False, 1)

    block3_1 = res_block_layers(block2_4, "block3_1", [256, 1024], True, 2)
    block3_2 = res_block_layers(block3_1, "block3_2", [256, 1024], False, 1)
    block3_3 = res_block_layers(block3_2, "block3_3", [256, 1024], False, 1)
    block3_4 = res_block_layers(block3_3, "block3_4", [256, 1024], False, 1)
    block3_5 = res_block_layers(block3_4, "block3_5", [256, 1024], False, 1)
    block3_6 = res_block_layers(block3_5, "block3_6", [256, 1024], False, 1)

    # block4_1 = res_block_layers(block3_6, "block4_1", [512, 2048], True, 2)
    # block4_2 = res_block_layers(block4_1, "block4_2", [512, 2048], False, 1)
    # block4_3 = res_block_layers(block4_2, "block4_3", [512, 2048], False, 1)

    # pool2 = avg_pool_op(block4_3, "pool2", kh=7, kw=7, dh=1, dw=1, padding="SAME")
    # shape = pool2.get_shape()
    # fc_in = tf.reshape(pool2, [-1, shape[1].value * shape[2].value * shape[3].value])
    # #logits, prob = fc_op(fc_in, "fc1", num_class, activation=tf.nn.softmax)
    # logits, prob = fc_op(fc_in, "fc1", num_class, activation=tf.nn.softmax)
    #2019/11/14/add
    #pool2 = avg_pool_op(block4_3, "pool2", kh=7, kw=7, dh=1, dw=1, padding="SAME")
    ######fc1
    # print(block4_3)
    # shape = block4_3.get_shape()
    # fc_in = tf.reshape(block4_3, [-1, shape[1].value * shape[2].value * shape[3].value])
    # logits, prob = fc_op(fc_in, "fc1", 1024, activation=tf.nn.softmax)
    # print(logits)
    #####fc2
    #shape = logits.get_shape()
    #print(shape)
    shape = block3_6.get_shape()
    fc_in = tf.reshape(block3_6, [-1, shape[1].value * shape[2].value * shape[3].value])
    logits, prob = fc_op(fc_in, "fc2", 512, activation=tf.nn.softmax)
    print(logits)
    #####fc3
    #shape = logits.get_shape()
    #fc_in = tf.reshape(logits, [-1, shape[1].value * shape[2].value * shape[3].value])
    logits, prob = fc_op(logits, "fc3", num_class, activation=tf.nn.softmax)
    print(logits)
    return logits, prob

