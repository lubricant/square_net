import re
import sys
import copy
import inspect
from functools import reduce
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from tensorflow.python.ops import init_ops


class _Dict(dict):
    pass


def __assert_shape(shape, *dim):
    assert isinstance(shape, (list, tuple, tf.TensorShape))
    assert all(x is None or isinstance(x, (int, tf.Dimension)) for x in shape)
    assert len(shape) in dim
    return True


def __assert_value(value, *args):
    assert value in args
    return True


def __assert_type(value, *args):
    assert isinstance(value, args)
    return True


def __attach_attr(obj, **args):
    for name in args:
        assert not hasattr(obj, name)
        setattr(obj, name, args[name])
    return obj


def __attach_ops(tensor, **args):
    op_list = _Dict()
    op_list.update(args)
    __attach_attr(op_list, **args)
    __attach_attr(tensor, ops=op_list)


def __attach_vars(tensor, **args):
    var_list = _Dict()
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
        return init_ops.variance_scaling_initializer(scale=2.)

    if mode == 'caffe':  # xavier with fan_in
        return init_ops.variance_scaling_initializer(scale=1., distribution='uniform')

    if mode == 'xavier':  # xavier with fan_avg
        return init_ops.glorot_normal_initializer()

    if mode.startswith('gauss'):
        stddev = gauss_with_opt.findall(mode)
        stddev = None if not stddev else float(stddev[0])
        if stddev:  # normal with 0 mean and specific stddev
            return init_ops.random_normal_initializer(stddev=stddev)
        else:  # np.random.randn(n) / sqrt(fan_in) [tanH]
            return init_ops.variance_scaling_initializer(scale=1)

    fan_opt = xavier_with_opt.findall(mode)
    fan_type = 'avg' if not fan_opt else fan_opt[0].lower()
    return init_ops.variance_scaling_initializer(scale=1., distribution='uniform', mode='fan_'+fan_type)


def data(name, shape, data_type=None):

    _ = _Scope.default_param(data)
    data_type = _(data_type, data_type=tf.float32)

    __assert_type(name, str)
    __assert_shape(shape, 1, 2, 4)  # [batch, height, width, channel] [batch, num_classes] [batch] = shape
    __assert_value(data_type, tf.float16, tf.float32, tf.float64, tf.int8, tf.int16, tf.int32, tf.int64)

    with tf.name_scope(name):
        return __attach_attr(tf.placeholder(data_type, shape, name='data'), naming=name)


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


def convolution(name, k_shape, stride=None, padding=None, random=None, active=None, batch_norm=None, training=None, data_type=None):

    __ = _Scope.default_param(convolution)
    stride = __(stride, stride=1)
    random = __(random, random='xavier')
    active = __(active, active=tf.nn.relu)
    padding = __(padding, padding='valid').upper()
    training = __(training, training=True)
    data_type = __(data_type, data_type=tf.float32)
    batch_norm = __(batch_norm, batch_norm=None)

    __assert_type(name, str)
    __assert_type(stride, int)
    __assert_value(active, tf.sigmoid, tf.tanh, tf.nn.relu, tf.nn.relu6, tf.nn.crelu, tf.nn.softplus, tf.nn.softsign)
    __assert_value(padding, 'VALID', 'SAME')
    __assert_shape(k_shape, 3)  # k_height, k_width, out_channel = k_shape

    def __(value):

        k_height, k_width, out_channel = k_shape
        in_channel = value.shape.as_list()[-1]

        with tf.variable_scope(name):
            shape = [k_height, k_width, in_channel, out_channel]

            filt = tf.get_variable('weight', shape, initializer=__initializer(random), trainable=training, dtype=data_type)
            conv = tf.nn.conv2d(value, filt, [1, stride, stride, 1], padding)

            if batch_norm:
                bias = None
                output = normalization('batch_norm', 'BATCH', training=training, data_type=data_type)(conv)
            else:
                bias = tf.get_variable('bias', conv.shape[-1:], initializer=__initializer(), trainable=training, dtype=data_type)
                output = tf.nn.bias_add(conv, bias)

            act = active(output)
            __attach_ops(act, conv=conv, active=act)
            __attach_vars(act, weight=filt, bias=bias)
            return __attach_attr(act, naming=name)

    return __


def pooling(name, p_shape, mode=None, stride=None, padding=None, training=None, data_type=None):

    __ = _Scope.default_param(pooling)
    mode = __(mode, mode='MAX').upper()
    stride = __(stride, stride=1)
    padding = __(padding, padding='valid').upper()
    training = __(training, training=True)
    data_type = __(data_type, data_type=tf.float32)

    __assert_type(name, str)
    __assert_type(mode, str)
    __assert_type(stride, int)
    __assert_value(mode, 'MAX', 'AVG')
    __assert_value(padding, 'VALID', 'SAME')
    __assert_shape(p_shape, 2, 3)  # p_height, p_width[, p_depth] = p_shape

    def __(value):

        # if not specific depth of pooling
        if len(p_shape):
            p_func = tf.nn.max_pool if mode == 'MAX' else tf.nn.avg_pool
            with tf.name_scope(name):
                pool = p_func(value, [1] + p_shape + [1], [1, stride, stride, 1], padding.upper(), name=mode.lower() + '_pool')
                __attach_ops(pool, conv=None, pool=pool)
                return __attach_attr(pool, naming=name, conv=None, pool=pool)

        else:  # if specific depth of pooling
            v_depth = value.shape.as_list()[-1]
            p_height, p_width, p_depth = p_shape
            assert p_depth == v_depth * 2

            with tf.variable_scope(name):
                conv = convolution('conv', [p_height, p_width, p_depth-v_depth], stride=stride, padding=padding, training=training, dtype=data_type)(value)
                pool = pooling('pool', [p_height, p_width], mode=mode, stride=stride, padding=padding, training=training)(value)
                concat = tf.concat([conv, pool], axis=-1, name='depth_concat')
                __attach_ops(concat, conv=None, pool=pool)
                return __attach_attr(concat, naming=name)

    return __


def density(name, neurons, linear=None, random=None, active=None, training=None, data_type=None):

    __ = _Scope.default_param(density)
    linear = __(linear, linear=False)
    random = __(random, random='gauss')
    active = __(active, active=tf.nn.relu)
    training = __(training, training=True)
    data_type = __(data_type, data_type=tf.float32)

    __assert_type(name, str)
    __assert_type(neurons, int)
    __assert_value(active, tf.sigmoid, tf.tanh, tf.nn.relu, tf.nn.relu6, tf.nn.crelu, tf.nn.softplus, tf.nn.softsign)

    def __(value):
        with tf.variable_scope(name):
            value = tf.reshape(value, [-1, np.prod(value.shape[1:].as_list())])

            shape = value.shape[1:].as_list() + [neurons]
            weight = tf.get_variable('weight', shape, initializer=__initializer(random), trainable=training, dtype=data_type)
            bias = tf.get_variable('bias', [neurons], initializer=__initializer(), trainable=training, dtype=data_type)
            fc = tf.nn.bias_add(tf.matmul(value, weight), bias)
            act = None if linear else active(fc)

            dense = fc if linear else act
            __attach_ops(dense, dense=fc, active=act)
            __attach_vars(dense, weight=weight, bias=bias)
            return __attach_attr(dense, naming=name)

    return __


def inception(name, *graph, training=None):

    assert graph
    assert all(isinstance(x, list) for x in graph)
    assert all(all(isinstance(x, tuple) and len(x) in (2, 3) for x in pipeline) for pipeline in graph)

    node_types = ('pool_3x3', 'conv_1x1', 'conv_3x3', 'conv_5x5', 'conv_1x7', 'conv_7x1')

    icp_args = _Scope.default_param(inception, as_dict=True)
    icp_args.update({'padding': 'same'})
    if training is not None:
        icp_args['training'] = training

    def __(value):

        with tf.variable_scope(name):

            types1 = reduce(lambda _nn, _branch: _nn + reduce(lambda _n, _node: _n + list(_node[0:1]), _branch, []), graph, [])
            types1 = set(map(lambda _key: _key if types1.count(_key) == 1 else None, node_types))

            b1_types = list(map(lambda _branch: _branch[0][0], filter(lambda _branch: len(_branch) == 1, graph)))
            b1_types = set(map(lambda _key: _key if b1_types.count(_key) == 1 and _key not in types1 else None, node_types))

            node_stack, node_path = [], []
            for branch in graph:
                node, path, is_b1 = value, [], len(branch) == 1

                for x in branch:

                    node_type, node_depth = x[0:2]
                    assert node_type in node_types

                    spec_args = {} if len(x) <= 2 else x[2]
                    scope_args = _Scope.default_param(getattr(inception, node_type), as_dict=True)

                    node_args = copy.deepcopy(icp_args)
                    node_args.update(scope_args)
                    node_args.update(spec_args)

                    node_alias = node_args['name'] if 'name' in node_args else None
                    if not node_alias:
                        if not is_b1 and node_type not in b1_types and node_type not in types1:
                            node_alias = node_type + '-' + str(node_depth)
                        else:
                            if node_type in b1_types:
                                b1_types.remove(node_type)
                            if node_type in types1:
                                types1.remove(node_type)

                    node = getattr(inception, node_type)(node_args, node_alias, node_depth, node)
                    path.append((node_type, node))
                node_stack.append(node)
                node_path.append(path)

            concat = tf.concat(node_stack, axis=-1, name='depth_concat')
            __attach_ops(concat, graph=tuple(node_path))
            return __attach_attr(concat, naming=name)

    return __

__attach_attr(inception, **{
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
            'pool_proj' if not alias else alias+'_proj', [1, 1, depth], **args)(pooling(
            'pool_3x3' if not alias else alias, [3, 3], 'MAX', padding='SAME')(value)),
})


def dropout(name, noise_shape=None):

    _ = _Scope.default_param(dropout)
    noise_shape = _(noise_shape, noise_shape=None)

    def __(value):
        with tf.name_scope(name):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            drop = tf.nn.dropout(value, name='dropout', keep_prob=keep_prob, noise_shape=noise_shape)

            __attach_vars(drop, keep_prob=keep_prob)
            return __attach_attr(drop, naming=name)
    return __


def normalization(name, mode=None, training=None, data_type=None,
                  batch_scale=None, batch_shift=None, batch_epsilon=None, batch_decay=None,
                  local_depth=None, local_bias=None, local_alpha=None, local_beta=None):

    __ = _Scope.default_param(normalization)
    mode = __(mode, mode='BATCH').upper()
    training = __(training, training=True)
    data_type = __(data_type, data_type=tf.float32)

    __assert_value(mode, 'LOCAL', 'BATCH')

    depth = __(local_depth, local_depth=5)
    bias = __(local_bias, local_bias=1.)
    alpha = __(local_alpha, local_alpha=1.)
    beta = __(local_beta, local_beta=.5)

    scaling = __(batch_scale, batch_scale=False)
    shifting = __(batch_shift, batch_shift=True)
    epsilon = __(batch_epsilon, batch_epsilon=0.001)
    decay = __(batch_decay, batch_decay=.999)

    def local_resp_norm(value):

        with tf.name_scope(name):
            norm = tf.nn.local_response_normalization(
                value, name='local_resp_norm', depth_radius=depth, bias=bias, alpha=alpha, beta=beta)

            __attach_ops(norm, norm=norm)
            return __attach_attr(norm, naming=name)

    def batch_norm(value):

        with tf.variable_scope(name):
            __assert_shape(value.get_shape(), 2, 4)  # density or conv2d

            shape = value.get_shape().as_list()
            channel = shape[-1]

            if len(shape) == 2:
                value = tf.reshape(value, [-1, 1, 1, channel], 'flatten')

            scale = tf.get_variable('gamma', channel, initializer=tf.ones_initializer(), trainable=training and scaling, dtype=data_type)
            shift = tf.get_variable('beta', channel, initializer=tf.zeros_initializer(), trainable=training and shifting, dtype=data_type)

            norm, mean, variance = tf.nn.fused_batch_norm(
                value, name='batch_norm', scale=scale, offset=shift, epsilon=epsilon, is_training=training)

            if not training:
                ema_mean = tf.get_variable('mean', channel, initializer=tf.zeros_initializer(), trainable=False, dtype=data_type)
                ema_variance = tf.get_variable('variance', channel, initializer=tf.ones_initializer(), trainable=False, dtype=data_type)

                from tensorflow.python.training import moving_averages as ema
                update_mean = ema.assign_moving_average(ema_mean, mean, decay=decay)
                update_variance = ema.assign_moving_average(ema_variance, variance, decay=decay)
                with tf.control_dependencies([update_mean, update_variance]):
                    norm = tf.identity(norm)

            if len(shape) == 2:
                norm = tf.reshape(norm, shape, 'stack')

            __attach_ops(norm, norm=norm)
            __attach_vars(norm, scale=scale, shift=shift)
            return __attach_attr(norm, naming=name)

    return local_resp_norm if mode.upper() == 'LOCAL' else batch_norm


@contextmanager
def default(layers, **params):

    if inspect.isfunction(layers):
        layers = [layers]
    elif layers == sys.modules[__name__]:
        layers = [*_Scope.scope_funcs.values()]

    assert __assert_type(layers, list, tuple)
    assert all(inspect.isfunction(x) for x in layers)
    assert all(_Scope.scope_funcs.values() or hasattr(inception, x.__name__) for x in layers)

    with _Scope(layers, params):
        yield


class _Scope(object):

    scope_funcs = dict(inspect.getmembers(sys.modules[__name__], lambda x: inspect.isfunction(x)))
    scope_stack = []

    @staticmethod
    def parse_param(layers, params):
        default_params = {}
        for layer in layers:
            if layer not in default_params:
                default_params[layer] = {}
            default_params[layer].update(params)
        return default_params

    @staticmethod
    def default_param(layer, as_dict=False):
        assert inspect.isfunction(layer)

        assert layer in _Scope.scope_funcs.values() or layer in vars(inception).values()

        parent_scope = None if not _Scope.scope_stack else _Scope.scope_stack[-1]
        parent_ctx = {} if not parent_scope else parent_scope.params_ctx

        merged_param = {}
        if layer in parent_ctx:
            merged_param.update(parent_ctx[layer])

        if as_dict:
            return merged_param

        def __(spec_val, **def_val):

            if spec_val is not None:
                return spec_val

            assert len(def_val) == 1
            item, = def_val.items()
            return merged_param.get(*item)

        return __

    def __init__(self, layers, params):
        self.params_ctx = None
        self.layers = layers
        self.params = params

    def __enter__(self):
        self.params_ctx = {}

        if _Scope.scope_stack:
            self.params_ctx = copy.deepcopy(_Scope.scope_stack[-1].params_ctx)

        for layer, params in _Scope.parse_param(self.layers, self.params).items():
            if layer not in self.params_ctx:
                self.params_ctx[layer] = params
            else:
                self.params_ctx[layer].update(params)

        _Scope.scope_stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _Scope.scope_stack.pop()
