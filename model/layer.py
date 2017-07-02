import numpy as np
import tensorflow as tf


def __assert_shape(shape, dim):
    assert isinstance(shape, (list, tuple)) and \
           all(x is None or isinstance(x, int) for x in shape) and \
           len(shape) == dim


def __assert_value(value, *args):
    assert value in args


def __assert_type(value, *args):
    assert isinstance(value, args)


def data(name, shape, elem=tf.float16):
    __assert_type(name, str)
    __assert_shape(shape, 4)  # batch, height, width, channel = shape
    __assert_value(elem, tf.float16, tf.float32, tf.float64, tf.int8, tf.int16, tf.int32, tf.int64)

    return lambda _=None: tf.placeholder(elem, shape, name)


def loss(name):
    tf.nn.softmax_cross_entropy_with_logits()
    tf.nn.softmax_cross_entropy_with_logits()
    pass


def convolution(name, k_shape, stride=1, padding='SAME'):
    __assert_type(name, str)
    __assert_type(stride, int)
    __assert_value(padding.upper(), 'VALID', 'SAME')
    __assert_shape(k_shape, 4)  # k_height, k_width, in_channel, out_channel = k_shape

    def __(value):
        with tf.variable_scope(name):
            rand = tf.random_normal(k_shape, stddev=1. / np.sqrt(np.sum(k_shape)))
            filt = tf.Variable(rand, name='filt')
            conv = tf.nn.conv2d(value, filt, [stride]*4, padding.upper())
            bias = tf.Variable(tf.zeros(conv.shape[-1]), name='bias')
            relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
            return relu

    return __


def pooling(name, p_shape, p_type, stride=1, padding='SAME'):
    __assert_type(name, str)
    __assert_type(p_type, str)
    __assert_type(stride, int)
    __assert_value(p_type.upper(), 'MAX', 'AVG')
    __assert_value(padding.upper(), 'VALID', 'SAME')
    __assert_shape(p_shape, 2)  # p_height, p_width = p_shape

    p_shape = [1] + p_shape + [1]
    stride = [1, stride, stride, 1]
    pool = tf.nn.max_pool if p_type.upper() == 'MAX' else tf.nn.avg_pool
    return lambda value: pool(value, p_shape, stride, padding.upper(), name=name)


def density(name, neurons):
    __assert_type(name, str)
    __assert_type(neurons, int)

    def __(value):
        with tf.variable_scope(name):
            value = tf.reshape(value, [-1, np.prod(value.shape[1:].as_list())])

            shape = value.shape[1:].as_list() + [neurons]
            rand = tf.random_normal(shape, stddev=1. / np.sqrt(np.sum(shape)))

            weight = tf.Variable(rand, name='weight')
            bias = tf.Variable(tf.zeros([neurons]), name='bias')
            relu = tf.nn.relu(tf.nn.bias_add(tf.matmul(value, weight), bias))
            return relu

    return __


def normalization(name):
    return lambda value: tf.nn.local_response_normalization(value, name=name)


def inception(name, reduce, depth):

    __assert_shape(reduce, 2)
    __assert_shape(depth, 4)

    reduce_3x3, reduce_5x5 = reduce
    depth_1x1, depth_3x3, depth_5x5, depth_pool = depth

    def __(value):

        __assert_shape(value.shape, 4)  # batch, height, width, channel = shape

        with tf.name_scope(name):
            v_depth = value.shape[-1]

            conv_1x1 = convolution('conv_1x1', [1, 1, v_depth, depth_1x1])(value)

            conv_reduce_3x3 = convolution('reduce_3x3', [1, 1, v_depth, reduce_3x3])(value)
            conv_3x3 = convolution('conv_3x3', [3, 3, reduce_3x3, depth_3x3])(conv_reduce_3x3)

            conv_reduce_5x5 = convolution('reduce_5x5', [1, 1, v_depth, reduce_5x5])(value)
            conv_5x5 = convolution('conv_5x5', [5, 5, reduce_5x5, depth_5x5])(conv_reduce_5x5)

            pool_3x3 = pooling('pool_3x3', [3, 3], 'MAX')(value)
            conv_pool_proj = convolution('pool_proj', [1, 1, v_depth, depth_pool])(pool_3x3)

            return tf.concat([conv_1x1, conv_3x3, conv_5x5, conv_pool_proj], axis=-1, name='depth_concat')
