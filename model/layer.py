import re
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc


from model.layer import *


class __Dict(dict):
    pass


def __assert_shape(shape, *dim):
    assert isinstance(shape, (list, tuple, tf.TensorShape))
    assert all(x is None or isinstance(x, (int, tf.Dimension)) for x in shape)
    assert len(shape) in dim


def __assert_value(value, *args):
    assert value in args


def __assert_type(value, *args):
    assert isinstance(value, args)


def __attach_attr(obj, **args):
    for name in args:
        assert not hasattr(obj, name)
        setattr(obj, name, args[name])
    return obj


def __attach_ops(tensor, **args):
    op_list = __Dict()
    op_list.update(args)
    __attach_attr(op_list, **args)
    __attach_attr(tensor, ops=args)


def __attach_vars(tensor, **args):
    var_list = __Dict()
    var_list.update(args)
    __attach_attr(var_list, **args)
    __attach_attr(tensor, vars=var_list)


def __initializer(mode=None):

    if not mode:
        return tf.zeros_initializer()

    gauss_with_opt = re.compile('^gauss:([-+]?[0-9]*\.?[0-9]+)$', re.I)
    xavier_with_opt = re.compile('^xavier:(in|out|avg)$', re.I)

    if not gauss_with_opt.match(mode) and not xavier_with_opt.match(mode):
        __assert_value(mode.lower(), 'gauss', 'xavier', 'caffe', 'msra')

    mode = mode.lower()

    if mode == 'msra':  # np.random.randn(n) * sqrt(2.0/fan_in) [ReLU]
        return tfc.layers.variance_scaling_initializer()

    if mode == 'caffe':  # xavier with fan_in
        return tfc.layers.variance_scaling_initializer(factor=1.0, uniform=True)

    if mode == 'xavier':  # xavier with fan_avg
        return tf.glorot_uniform_initializer()

    if mode.startswith('gauss'):
        stddev = gauss_with_opt.findall(mode)
        stddev = None if not stddev else float(stddev[0])
        if stddev:  # normal with 0 mean and specific stddev
            return tf.truncated_normal_initializer(stddev=stddev)
        else:  # np.random.randn(n) / sqrt(fan_in) [tanH]
            return tfc.layers.variance_scaling_initializer(factor=1.0)

    fan_opt = xavier_with_opt.findall(mode)
    fan_type = 'AVG' if not fan_opt else fan_opt[0].upper()
    return tfc.layers.variance_scaling_initializer(factor=1.0, uniform=True, mode='FAN_'+fan_type)


def data(name, shape, dtype=tf.float32):
    __assert_type(name, str)
    __assert_shape(shape, 1, 2, 4)  # [batch, height, width, channel] [batch, num_classes] [batch] = shape
    __assert_value(dtype, tf.float16, tf.float32, tf.float64, tf.int8, tf.int16, tf.int32, tf.int64)

    with tf.name_scope(name):
        return __attach_attr(tf.placeholder(dtype, shape, name='data'), naming=name)


def loss(name):

    def __(logits, labels):
        __assert_shape(logits.shape.as_list(), 2)  # batch_size, num_classes = logits.shape
        __assert_shape(labels.shape.as_list(), 1, 2)

        l_func = (tf.nn.softmax_cross_entropy_with_logits  # batch_size, num_classes = labels.shape
                  if len(labels.shape) == 2 else
                  tf.nn.sparse_softmax_cross_entropy_with_logits)  # batch_size = labels.shape

        with tf.name_scope(name):
            cross_entropy = l_func(logits=logits, labels=labels, name='cross_entropy')
            return __attach_attr(tf.reduce_mean(cross_entropy), naming=name)

    return __


def convolution(name, k_shape, stride=1, padding='VALID', random='xavier', training=True):
    __assert_type(name, str)
    __assert_type(stride, int)
    __assert_value(padding.upper(), 'VALID', 'SAME')
    __assert_shape(k_shape, 3)  # k_height, k_width, out_channel = k_shape

    def __(value):

        k_height, k_width, out_channel = k_shape
        in_channel = value.shape.as_list()[-1]

        with tf.variable_scope(name):
            shape = [k_height, k_width, in_channel, out_channel]

            filt = tf.get_variable('weight', shape, initializer=__initializer(random), trainable=training)
            conv = tf.nn.conv2d(value, filt, [1, stride, stride, 1], padding.upper())
            bias = tf.get_variable('bias', conv.shape[-1:], initializer=__initializer(), trainable=training)
            relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
            __attach_ops(relu, conv=conv, active=relu)
            __attach_vars(relu, weight=filt, bias=bias)
            return __attach_attr(relu, naming=name)

    return __


def pooling(name, p_shape, p_type, stride=1, padding='VALID', training=True):
    __assert_type(name, str)
    __assert_type(p_type, str)
    __assert_type(stride, int)
    __assert_value(p_type.upper(), 'MAX', 'AVG')
    __assert_value(padding.upper(), 'VALID', 'SAME')
    __assert_shape(p_shape, 2, 3)  # p_height, p_width[, p_depth] = p_shape

    def __(value):

        # if not specific depth of pooling
        if len(p_shape):
            p_func = tf.nn.max_pool if p_type.upper() == 'MAX' else tf.nn.avg_pool
            with tf.name_scope(name):
                pool = p_func(value, [1] + p_shape + [1], [1, stride, stride, 1], padding.upper(), name=p_type.lower()+'_pool')
                __attach_ops(pool, conv=None, pool=pool)
                return __attach_attr(pool, naming=name, conv=None, pool=pool)

        else:  # if specific depth of pooling
            v_depth = value.shape.as_list()[-1]
            p_height, p_width, p_depth = p_shape
            assert p_depth == v_depth * 2

            with tf.variable_scope(name):
                conv = convolution('conv', [p_height, p_width, p_depth-v_depth], stride=stride, padding=padding, training=training)(value)
                pool = pooling('pool', [p_height, p_width], p_type=p_type, stride=stride, padding=padding, training=training)(value)
                concat = tf.concat([conv, pool], axis=-1, name='depth_concat')
                __attach_ops(concat, conv=None, pool=pool)
                return __attach_attr(concat, naming=name)

    return __


def density(name, neurons, linear=False, random='gauss', training=True):
    __assert_type(name, str)
    __assert_type(neurons, int)

    def __(value):
        with tf.variable_scope(name):
            value = tf.reshape(value, [-1, np.prod(value.shape[1:].as_list())])

            shape = value.shape[1:].as_list() + [neurons]
            weight = tf.get_variable('weight', shape, initializer=__initializer(random), trainable=training)
            bias = tf.get_variable('bias', [neurons], initializer=__initializer(), trainable=training)
            fc = tf.nn.bias_add(tf.matmul(value, weight), bias)
            act = None if linear else tf.nn.relu(fc)

            dense = fc if linear else act
            __attach_ops(dense, dense=fc, active=act)
            __attach_vars(dense, weight=weight, bias=bias)
            return __attach_attr(dense, naming=name)

    return __


def inception(name, *graph, training=True):
    assert graph and \
           all(isinstance(x, list) for x in graph) and \
           all(all(isinstance(x, tuple) and len(x) in (2, 3) for x in pipeline) for pipeline in graph)

    node_factory = {
        'conv_1x1':
            lambda args, alias, depth, value: convolution(
                'conv_1x1' if not alias else alias, [1, 1, depth], **args)(value),
        'conv_3x3':
            lambda args, alias, depth, value: convolution(
                'conv_3x3' if not alias else alias, [3, 3, depth], **args)(value),
        'conv_5x5':
            lambda args, alias, depth, value: convolution(
                'conv_5x5' if not alias else alias, [5, 5, depth], **args)(value),
        'conv_1x7':
            lambda args, alias, depth, value: convolution(
                'conv_1x7' if not alias else alias, [1, 7, depth], **args)(value),
        'conv_7x1':
            lambda args, alias, depth, value: convolution(
                'conv_7x1' if not alias else alias, [7, 1, depth], **args)(value),
        'pool_3x3':
            lambda args, alias, depth, value: convolution(
                'pool_proj' if not alias else alias, [1, 1, depth], **args)(pooling(
                'pool_3x3' if not alias else alias+'_'+'pool_3x3', [3, 3], 'MAX', padding='SAME')(value)),
    }

    def __(value):

        with tf.variable_scope(name):

            default_args = {'padding': 'same', 'random': 'xavier', 'training': training}

            types1 = reduce(lambda _nn, _branch: _nn + reduce(lambda _n, _node: _n + list(_node[0:1]), _branch, []), graph, [])
            types1 = set(map(lambda _key: _key if types1.count(_key) == 1 else None, node_factory.keys()))

            b1_types = list(map(lambda _branch: _branch[0][0], filter(lambda _branch: len(_branch) == 1, graph)))
            b1_types = set(map(lambda _key: _key if b1_types.count(_key) == 1 else None, node_factory.keys()))

            node_stack, node_path = [], []
            for branch in graph:
                node, path, is_b1 = value, [], len(branch) == 1
                for x in branch:
                    node_type, node_depth = x[0:2]
                    assert node_type in node_factory

                    node_args = default_args if len(x) <= 2 else x[2]
                    node_args['training'] = training

                    node_alias = node_args['name'] if 'name' in node_args else None
                    if not node_alias:
                        if not is_b1 and node_type not in b1_types and node_type not in types1:
                            node_alias = node_type + '-' + str(node_depth)

                    node = node_factory[node_type](node_args, node_alias, node_depth, node)
                    path.append((node_type, node))
                node_stack.append(node)
                node_path.append(path)

            concat = tf.concat(node_stack, axis=-1, name='depth_concat')
            __attach_ops(concat, graph=tuple(node_path))
            return __attach_attr(concat, naming=name)

    return __


def dropout(name, **args):
    def __(value):
        with tf.name_scope(name):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            drop = tf.nn.dropout(value, name='dropout', keep_prob=keep_prob, **args)

            __attach_vars(drop, keep_prob=keep_prob)
            return __attach_attr(drop, naming=name)


def normalization(name, mode, training=True,
                  batch_scale=False, batch_shift=True, batch_epsilon=0.001, batch_decay=.999,
                  local_depth=5, local_bias=1., local_alpha=1., local_beta=.5):

    __assert_value(mode.upper(), 'LOCAL', 'BATCH')

    def local_resp_norm(value):
        with tf.name_scope(name):
            norm = tf.nn.local_response_normalization(
                value, name='local_resp_norm',
                depth_radius=local_depth, bias=local_bias, alpha=local_alpha, beta=local_beta)

            __attach_ops(norm, norm=norm)
            return __attach_attr(norm, naming=name)

    def batch_norm(value):

        with tf.variable_scope(name):
            __assert_shape(value.get_shape(), 2, 4)  # density or conv2d

            shape = value.get_shape().as_list()
            channel = shape[-1]

            if len(shape) == 2:
                value = tf.reshape(value, [-1, 1, 1, channel], 'flatten')

            scale = tf.get_variable('gamma', channel, initializer=tf.ones_initializer(), trainable=training and batch_scale)
            shift = tf.get_variable('beta', channel, initializer=tf.zeros_initializer(), trainable=training and batch_shift)

            norm, mean, variance = tf.nn.fused_batch_norm(
                value, name='batch_norm', scale=scale, offset=shift, epsilon=batch_epsilon, is_training=training)

            if not training:
                ema_mean = tf.get_variable('mean', channel, initializer=tf.zeros_initializer(), trainable=False)
                ema_variance = tf.get_variable('variance', channel, initializer=tf.ones_initializer(), trainable=False)

                from tensorflow.python.training import moving_averages as ema
                update_mean = ema.assign_moving_average(ema_mean, mean, decay=batch_decay)
                update_variance = ema.assign_moving_average(ema_variance, variance, decay=batch_decay)
                with tf.control_dependencies([update_mean, update_variance]):
                    norm = tf.identity(norm)

            if len(shape) == 2:
                norm = tf.reshape(norm, shape, 'stack')

            __attach_ops(norm, norm=norm)
            __attach_vars(norm, scale=scale, shift=shift)
            return __attach_attr(norm, naming=name)

    return local_resp_norm if mode.upper() == 'LOCAL' else batch_norm

