import numpy as np
import tensorflow as tf


def __assert_shape(shape, *dim):
    assert isinstance(shape, (list, tuple)) and \
           all(x is None or isinstance(x, int) for x in shape) and \
           len(shape) in dim


def __assert_value(value, *args):
    assert value in args


def __assert_type(value, *args):
    assert isinstance(value, args)


def data(name, shape, dtype=tf.float32):
    __assert_type(name, str)
    __assert_shape(shape, 1, 2, 4)  # [batch, height, width, channel] [batch, num_classes] [batch] = shape
    __assert_value(dtype, tf.float16, tf.float32, tf.float64, tf.int8, tf.int16, tf.int32, tf.int64)

    return tf.placeholder(dtype, shape, name)


def loss(name):

    def __(logits, labels):
        __assert_shape(logits.shape.as_list(), 2)  # batch_size, num_classes = logits.shape
        __assert_shape(labels.shape.as_list(), 1, 2)

        l_func = (tf.nn.softmax_cross_entropy_with_logits  # batch_size, num_classes = labels.shape
                  if len(labels.shape) == 2 else
                  tf.nn.sparse_softmax_cross_entropy_with_logits)  # batch_size = labels.shape

        return l_func(logits=logits, labels=labels, name=name)

    return __


def convolution(name, k_shape, stride=1, padding='SAME', dtype=tf.float32):
    __assert_type(name, str)
    __assert_type(stride, int)
    __assert_value(padding.upper(), 'VALID', 'SAME')
    __assert_shape(k_shape, 3)  # k_height, k_width, out_channel = k_shape

    def __(value):

        k_height, k_width, out_channel = k_shape
        in_channel = value.shape.as_list()[-1]

        with tf.variable_scope(name):
            shape = [k_height, k_width, in_channel, out_channel]
            fan_in, fan_out = np.prod(shape[:-1]), np.prod(k_shape)
            bound = np.sqrt(6. / (fan_in + fan_out))
            rand = tf.random_uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

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
    __assert_shape(p_shape, 2, 3)  # p_height, p_width[, p_depth] = p_shape

    if len(p_shape) == 2:
        p_shape = [1] + p_shape + [1]
        stride = [1, stride, stride, 1]
        p_func = tf.nn.max_pool if p_type.upper() == 'MAX' else tf.nn.avg_pool
        return lambda value: p_func(value, p_shape, stride, padding.upper(), name=name)

    def __(value):
        v_depth = value.shape.as_list()[-1]
        p_height, p_width, p_depth = p_shape
        assert p_depth == v_depth * 2

        with tf.variable_scope(name):
            conv = convolution('conv', [p_height, p_width, p_depth-v_depth],
                               stride=stride, padding=padding)(value)
            pool = pooling('pool', [p_height, p_width], p_type=p_type,
                           stride=stride, padding=padding)(value)
            return tf.concat([conv, pool], axis=-1, name='depth_concat')

    return __


def density(name, neurons, dtype=tf.float32):
    __assert_type(name, str)
    __assert_type(neurons, int)

    def __(value):
        with tf.variable_scope(name):
            value = tf.reshape(value, [-1, np.prod(value.shape[1:].as_list())])

            shape = value.shape[1:].as_list() + [neurons]
            fan_in, fan_out = value.shape.as_list()[-1], neurons
            bound = np.sqrt(6. / (fan_in + fan_out))
            rand = tf.random_uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

            weight = tf.Variable(rand, name='weight')
            bias = tf.Variable(tf.zeros([neurons]), name='bias')
            relu = tf.nn.relu(tf.nn.bias_add(tf.matmul(value, weight), bias))
            return relu

    return __


def normalization(name):
    return lambda value: tf.nn.local_response_normalization(value, name=name)


def inception(name, *graph, dtype=tf.float32):
    assert graph and \
           all(isinstance(x, list) for x in graph) and \
           all(all(isinstance(x, tuple) and len(x) == 2 for x in pipeline) for pipeline in graph)

    node_factory = {
        'conv_1x1':
            lambda depth, value: convolution('conv_1x1', [3, 3, depth], padding='SAME', dtype=dtype)(value),
        'conv_3x3':
            lambda depth, value: convolution('conv_3x3', [5, 5, depth], padding='SAME', dtype=dtype)(value),
        'conv_5x5':
            lambda depth, value: convolution('conv_1x1', [5, 5, depth], padding='SAME', dtype=dtype)(value),
        'conv_1x7':
            lambda depth, value: convolution('conv_1x7', [1, 7, depth], padding='SAME', dtype=dtype)(value),
        'conv_7x1':
            lambda depth, value: convolution('conv_7x1', [7, 1, depth], padding='SAME', dtype=dtype)(value),
        'pool_3x3':
            lambda depth, value: pooling('pool_3x3', [3, 3], 'MAX')(
                convolution('pool_proj', [1, 1, depth], padding='SAME', dtype=dtype)(value)),
    }

    def __(value):

        with tf.name_scope(name):

            node_stack = []
            for pipeline in graph:
                node = value
                for node_type, node_depth in pipeline:
                    assert node_type in node_factory
                    node = node_factory[node_type](node_depth, node)
                node_stack.append(node)

            return tf.concat(node_stack, axis=-1, name='depth_concat')

    return __

# def inception(name, reduce, depth):
#
#     __assert_shape(reduce, 2)
#     __assert_shape(depth, 4)
#
#     reduce_3x3, reduce_5x5 = reduce
#     depth_1x1, depth_3x3, depth_5x5, depth_pool = depth
#
#     def __(value):
#
#         __assert_shape(value.shape, 4)  # batch, height, width, channel = shape
#
#         ('conv_1x1', 'conv_3x3', 'conv_5x5', 'conv_7x7', 'pool_3x3')
#
#         with tf.name_scope(name):
#             v_depth = value.shape[-1]
#
#             conv_1x1 = convolution('conv_1x1', [1, 1, v_depth, depth_1x1])(value)
#
#             conv_reduce_3x3 = convolution('reduce_3x3', [1, 1, v_depth, reduce_3x3])(value)
#             conv_3x3 = convolution('conv_3x3', [3, 3, reduce_3x3, depth_3x3])(conv_reduce_3x3)
#
#             conv_reduce_5x5 = convolution('reduce_5x5', [1, 1, v_depth, reduce_5x5])(value)
#             conv_5x5 = convolution('conv_5x5', [5, 5, reduce_5x5, depth_5x5])(conv_reduce_5x5)
#
#             pool_3x3 = pooling('pool_3x3', [3, 3], 'MAX')(value)
#             conv_pool_proj = convolution('pool_proj', [1, 1, v_depth, depth_pool])(pool_3x3)
#
#             return tf.concat([conv_1x1, conv_3x3, conv_5x5, conv_pool_proj], axis=-1, name='depth_concat')
