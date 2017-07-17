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


def __attach_attr(obj, **args):
    for name in args:
        assert not hasattr(obj, name)
        setattr(obj, name, args[name])
    return obj


def __random_init(shape, mode):

    __assert_shape(shape, 2, 4)
    __assert_value(mode.lower(), 'gauss', 'xavier', 'caffe', 'msra')

    mode = mode.lower()

    if len(shape) == 2:  # density
        fan_in, fan_out = shape
    else:  # convolution
        k_size = np.prod(shape[:2])
        fan_in = float(k_size * shape[2])
        fan_out = float(k_size * shape[3])

    if mode == 'gauss':
        with tf.name_scope('gauss_init'):
            return tf.truncated_normal(shape, 0.0, 0.01, tf.float32)

    if mode == 'msra':
        stddev = 2.0 / fan_in
        trunc_stddev = np.sqrt(1.3 * stddev)
        with tf.name_scope('msra_init'):
            return tf.truncated_normal(shape, 0.0, trunc_stddev, tf.float32)

    n = fan_in if (mode == 'caffe') else ((fan_in + fan_out) / 2.)
    limit = np.sqrt(3.0 / n)

    with tf.name_scope('%s_init' % mode):
        return tf.random_uniform(shape, -limit, limit, tf.float32)


def data(name, shape, dtype=tf.float32):
    __assert_type(name, str)
    __assert_shape(shape, 1, 2, 4)  # [batch, height, width, channel] [batch, num_classes] [batch] = shape
    __assert_value(dtype, tf.float16, tf.float32, tf.float64, tf.int8, tf.int16, tf.int32, tf.int64)

    return __attach_attr(tf.placeholder(dtype, shape, name), layer_name=name)


def loss(name):

    def __(logits, labels):
        __assert_shape(logits.shape.as_list(), 2)  # batch_size, num_classes = logits.shape
        __assert_shape(labels.shape.as_list(), 1, 2)

        l_func = (tf.nn.softmax_cross_entropy_with_logits  # batch_size, num_classes = labels.shape
                  if len(labels.shape) == 2 else
                  tf.nn.sparse_softmax_cross_entropy_with_logits)  # batch_size = labels.shape

        cross_entropy = l_func(logits=logits, labels=labels, name=name)
        return __attach_attr(tf.reduce_mean(cross_entropy), layer_name=name, cross_entropy=cross_entropy)

    return __


def convolution(name, k_shape, stride=1, padding='SAME', init='xavier'):
    __assert_type(name, str)
    __assert_type(stride, int)
    __assert_value(padding.upper(), 'VALID', 'SAME')
    __assert_shape(k_shape, 3)  # k_height, k_width, out_channel = k_shape

    def __(value):

        k_height, k_width, out_channel = k_shape
        in_channel = value.shape.as_list()[-1]

        with tf.name_scope(name):
            shape = [k_height, k_width, in_channel, out_channel]

            filt = tf.Variable(__random_init(shape, init), name='filt')
            conv = tf.nn.conv2d(value, filt, [1, stride, stride, 1], padding.upper())
            bias = tf.Variable(tf.zeros(conv.shape[-1]), name='bias')
            relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
            return __attach_attr(relu, layer_name=name, weight=filt, bias=bias)

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
        return lambda value: __attach_attr(
            p_func(value, p_shape, stride, padding.upper(), name=name), layer_name=name, weight=None, bias=None)

    def __(value):
        v_depth = value.shape.as_list()[-1]
        p_height, p_width, p_depth = p_shape
        assert p_depth == v_depth * 2

        with tf.name_scope(name):
            conv = convolution('conv', [p_height, p_width, p_depth-v_depth],
                               stride=stride, padding=padding)(value)
            pool = pooling('pool', [p_height, p_width], p_type=p_type,
                           stride=stride, padding=padding)(value)
            return __attach_attr(
                tf.concat([conv, pool], axis=-1, name='depth_concat'), layer_name=name, weight=conv.weight, bias=conv.bias)

    return __


def density(name, neurons, linear=False, init='xavier'):
    __assert_type(name, str)
    __assert_type(neurons, int)

    def __(value):
        with tf.name_scope(name):
            value = tf.reshape(value, [-1, np.prod(value.shape[1:].as_list())])

            shape = value.shape[1:].as_list() + [neurons]
            weight = tf.Variable(__random_init(shape, init), name='weight')
            bias = tf.Variable(tf.zeros([neurons]), name='bias')
            fc = tf.nn.bias_add(tf.matmul(value, weight), bias)
            return __attach_attr(fc if linear else tf.nn.relu(fc), layer_name=name, weight=weight, bias=bias)

    return __


def normalization(name):
    return lambda value: __attach_attr(tf.nn.local_response_normalization(value, name=name), layer=name)


def dropout(name, **args):
    with tf.name_scope(name):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return lambda value: __attach_attr(
            tf.nn.dropout(value, name='dropout', keep_prob=keep_prob, **args), layer_name=name, keep_prob=keep_prob)


def inception(name, *graph):
    assert graph and \
           all(isinstance(x, list) for x in graph) and \
           all(all(isinstance(x, tuple) and len(x) in (2, 3) for x in pipeline) for pipeline in graph)

    node_factory = {
        'conv_1x1':
            lambda init, depth, value: convolution('conv_1x1', [1, 1, depth], padding='SAME', init=init)(value),
        'conv_3x3':
            lambda init, depth, value: convolution('conv_3x3', [3, 3, depth], padding='SAME', init=init)(value),
        'conv_5x5':
            lambda init, depth, value: convolution('conv_5x5', [5, 5, depth], padding='SAME', init=init)(value),
        'conv_1x7':
            lambda init, depth, value: convolution('conv_1x7', [1, 7, depth], padding='SAME', init=init)(value),
        'conv_7x1':
            lambda init, depth, value: convolution('conv_7x1', [7, 1, depth], padding='SAME', init=init)(value),
        'pool_3x3':
            lambda init, depth, value: convolution('pool_proj', [1, 1, depth], padding='SAME', init=init)(
                                           pooling('pool_3x3', [3, 3], 'MAX')(value)),
    }

    def __(value):

        with tf.name_scope(name):

            node_stack, node_path = [], []
            for pipeline in graph:
                node, path = value, []
                for x in pipeline:
                    node_type, node_depth = x[0:2]
                    node_init = 'xavier' if len(x) <= 2 else x[2]
                    assert node_type in node_factory
                    node = node_factory[node_type](node_init, node_depth, node)
                    path.append((node_type, node))
                node_stack.append(node)
                node_path.append(path)

            return __attach_attr(tf.concat(node_stack, axis=-1, name='depth_concat'),
                                 layer_name=name, branch_graph=tuple(node_path))

    return __
