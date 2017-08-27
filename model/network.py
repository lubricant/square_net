
import tensorflow as tf
import model.layer as layer


class Net(object):

    @staticmethod
    def _show_scalar(**args):
        for name in args:
            tf.summary.scalar(name, args[name])

    @staticmethod
    def _show_tensor(loss, **args):
        for name, val in args.items():
            if val is None:
                continue
            with tf.name_scope(name):
                mean = tf.reduce_mean(val)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(val - mean)))

                grad = tf.gradients(loss, val)[0]
                grad_mean = tf.reduce_mean(grad)

                tf.summary.scalar('grad', grad_mean)
                tf.summary.scalar('mean', mean)
                tf.summary.scalar('stddev', stddev)
                tf.summary.histogram('hist', val)

    @staticmethod
    def _show_grad(loss, **args):
        for name, val in args.items():
            if val is None:
                continue
            with tf.name_scope(name):
                grad = tf.gradients(loss, val)[0]
                grad_mean = tf.reduce_mean(grad)
                tf.summary.scalar('grad', grad_mean)

    @staticmethod
    def _show_weight_and_bias(show_fn, *args):
        for var in args:
            with tf.name_scope(var.naming):
                weight, bias = var.vars.weight, var.vars.bias
                if isinstance(weight, dict):
                    with tf.name_scope('weight'):
                        show_fn(**weight)
                else:
                    assert isinstance(weight, tf.Variable)
                    show_fn(weight=weight)

                if isinstance(bias, dict):
                    with tf.name_scope('bias'):
                        show_fn(**bias)
                else:
                    assert bias is None or isinstance(bias, tf.Variable)
                    show_fn(bias=bias)

    @staticmethod
    def _show_branch_graph(show_fn, *args):
        for var in args:
            with tf.name_scope(var.naming):
                for branch in var.ops.graph:
                    if len(branch) == 1:
                        _, branch_node = branch[0]
                        Net._show_weight_and_bias(show_fn, branch_node)
                    else:
                        branch_name = '-'.join(map(lambda x: x[0], branch))
                        with tf.name_scope(branch_name):
                            Net._show_weight_and_bias(show_fn, *map(lambda x: x[1], branch))


class HCCR_GoogLeNet(Net):

    def __init__(self, is_training=True):
        with tf.variable_scope('HCCR-GoogLeNet'):
            with layer.default(layer, training=is_training):
                self.__build_network()

        self.__build_summary()

    def __build_network(self):

        FLAGS = tf.app.flags.FLAGS

        assert FLAGS.image_size <= 120
        assert FLAGS.label_num == 3755

        self.images = layer.data('Input', [None, 120, 120, FLAGS.image_channel])
        self.labels = layer.data('Label', [None], tf.int64)

        with layer.default([layer.normalization], mode='LOCAL', local_alpha=0.0001, local_beta=0.75),\
             layer.default([layer.pooling], mode='MAX', stride=2):

            self.conv1 = layer.convolution('Conv_7x7x64', [7, 7, 64], stride=2, random='gauss:0.015')(self.images)
            self.pool1 = layer.pooling('MaxPool_3x3', [3, 3])(self.conv1)
            self.norm1 = layer.normalization('LRN_1')(self.pool1)

            self.conv2 = layer.convolution('Conv_1x1x64', [1, 1, 64])(self.norm1)
            self.conv3 = layer.convolution('Conv_3x3x192', [3, 3, 192], padding='SAME', random='gauss:0.02')(self.conv2)
            self.norm2 = layer.normalization('LRN_2')(self.conv3)
            self.pool2 = layer.pooling('MaxPool_3x3', [3, 3])(self.norm2)

        with layer.default([layer.inception.conv_3x3], random='gauss:0.04', padding='same'),\
             layer.default([layer.inception.conv_5x5], random='gauss:0.08', padding='same'):

            self.incp1 = layer.inception('Inception_v1_1',
                                         [('pool_3x3', 32)],
                                         [('conv_1x1', 64)],
                                         [('conv_1x1', 96), ('conv_3x3', 128)],
                                         [('conv_1x1', 16), ('conv_5x5', 32)])(self.pool2)

            self.incp2 = layer.inception('Inception_v1_2',
                                         [('pool_3x3', 64)],
                                         [('conv_1x1', 128)],
                                         [('conv_1x1', 128), ('conv_3x3', 192)],
                                         [('conv_1x1', 32), ('conv_5x5', 96)])(
                                        self.incp1)

            self.pool3 = layer.pooling('MaxPool_3x3', [3, 3], stride=2)(self.incp2)

            self.incp3 = layer.inception('Inception_v1_3',
                                         [('pool_3x3', 64)],
                                         [('conv_1x1', 160)],
                                         [('conv_1x1', 112), ('conv_3x3', 224)],
                                         [('conv_1x1', 24), ('conv_5x5', 64)])(
                                         self.pool3)

            self.incp4 = layer.inception('Inception_v1_4',
                                         [('pool_3x3', 128)],
                                         [('conv_1x1', 256)],
                                         [('conv_1x1', 160), ('conv_3x3', 320)],
                                         [('conv_1x1', 32), ('conv_5x5', 128)])(
                                         self.incp3)

            self.pool4 = layer.pooling('AvgPool_5x5', [5, 5], 'AVG')(self.incp4)
            self.conv4 = layer.convolution('Conv_1x1x256', [1, 1, 256])(self.pool4)

        with layer.default([layer.density], random='gauss:0.01'):
            self.fc = layer.density('FC_1024', 1024)(self.conv4)
            self.dropout = layer.dropout('Dropout')(self.fc)
            self.logits = layer.density('FC_3755', 3755, linear=True)(self.dropout)
            self.keep_prob = self.dropout.vars.keep_prob

        self.loss = layer.loss('Loss')(self.logits, self.labels)

    def __build_summary(self):

        def show_tensor(**args):
            Net._show_tensor(self.loss, **args)

        def show_grad(**args):
            Net._show_grad(self.loss, **args)

        with tf.name_scope('Performance'):
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(self.labels, tf.argmax(self.logits, 1)), tf.float32))
            Net._show_scalar(loss=self.loss, accuracy=self.accuracy)

        with tf.name_scope('Convolution'):
            Net._show_weight_and_bias(show_tensor, self.conv1, self.conv2, self.conv3, self.conv4)

        with tf.name_scope('Inception'):
            Net._show_branch_graph(show_tensor, self.incp1, self.incp2, self.incp3, self.incp4)

        with tf.name_scope('FullyConnected'):
            Net._show_weight_and_bias(show_tensor, self.fc, self.logits)

        with tf.name_scope('Gradient'):
            Net._show_weight_and_bias(show_grad, self.conv1, self.conv2, self.conv3, self.conv4)
            Net._show_branch_graph(show_grad, self.incp1, self.incp2, self.incp3, self.incp4)
            Net._show_weight_and_bias(show_grad, self.fc, self.logits)


class MobileNet(Net):

    def __init__(self, is_training=True):
        with tf.variable_scope('SDD-MobileNet'):
            with layer.default(layer, training=is_training), \
                 layer.default([layer.convolution], mode='DEPTH_SEP', batch_norm=True), \
                 layer.default([layer.normalization], batch_shift=True, batch_scale=True, batch_decay=0.9997):
                self.__build_network()

        self.__build_summary()

    def __build_network(self):
        from functools import reduce
        FLAGS = tf.app.flags.FLAGS

        self.conv_ds = {}

        self.images = layer.data('Input', [None, 224, 224, 1])
        self.labels = layer.data('Label', [None], tf.int64)

        self.conv1 = layer.convolution('Conv_3x3', [3, 3, 32], stride=2, mode='STANDARD')(self.images)

        self.conv_ds_64 = layer.convolution('Conv_DS_64', [3, 3, 64], stride=1)(self.conv1)
        self.conv_ds_128 = layer.convolution('Conv_DS_128', [3, 3, 128], stride=2)(self.conv_ds_64)
        self.conv_ds_256 = layer.convolution('Conv_DS_256', [3, 3, 256], stride=1)(self.conv_ds_128)
        self.conv_ds_512 = layer.convolution('Conv_DS_512', [3, 3, 512], stride=2)(self.conv_ds_256)

        conv_ds_512 = [
            layer.convolution('Conv_DS_512_%d' % (i+1), [3, 3, 512], stride=1, padding='SAME')
            for i in range(5)
        ]
        conv_ds_1024 = [
            layer.convolution('Conv_DS_1024_%d' % (i+1), [3, 3, 1024], stride=2-i)
            for i in range(2)
        ]

        (
         self.conv_ds_512_1, self.conv_ds_512_2, self.conv_ds_512_3, self.conv_ds_512_4, self.conv_ds_512_5,
         self.conv_ds_1024_1, self.conv_ds_1024_2
        ) = reduce(lambda c_val, c_def: c_val + [c_def(c_val[-1])], conv_ds_512 + conv_ds_1024, [self.conv_ds_512])[1:]

        self.pool = layer.pooling('AvgPool_7x7', [7, 7], 'AVG')(self.conv_ds_1024_2)
        self.fc = layer.density('FC_1024', 1024)(self.pool)
        self.logits = layer.density('FC_2', 2, linear=True)(self.fc)
        self.loss = layer.loss('Loss')(self.logits, self.labels)

    def __build_summary(self):

        def show_tensor(**args):
            Net._show_tensor(self.loss, **args)

        def show_grad(**args):
            Net._show_grad(self.loss, **args)

        with tf.name_scope('Performance'):
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(self.labels, tf.argmax(self.logits, 1)), tf.float32))
            Net._show_scalar(loss=self.loss, accuracy=self.accuracy)

        with tf.name_scope('Convolution'):
            Net._show_weight_and_bias(show_tensor, self.conv1)
            Net._show_weight_and_bias(show_grad, self.conv1)

        with tf.name_scope('FullyConnection'):
            Net._show_weight_and_bias(show_tensor, self.fc, self.logits)
            Net._show_weight_and_bias(show_grad, self.fc, self.logits)

        with tf.name_scope('DepthSep'):
            Net._show_weight_and_bias(show_tensor, self.conv_ds_64, self.conv_ds_128, self.conv_ds_256)
            Net._show_weight_and_bias(show_grad, self.conv_ds_64, self.conv_ds_128, self.conv_ds_256)

        with tf.name_scope('DepthSep_512'):
            Net._show_weight_and_bias(
                show_tensor, self.conv_ds_512,
                self.conv_ds_512_1, self.conv_ds_512_2, self.conv_ds_512_3, self.conv_ds_512_4, self.conv_ds_512_5)
            Net._show_weight_and_bias(
                show_grad, self.conv_ds_512,
                self.conv_ds_512_1, self.conv_ds_512_2, self.conv_ds_512_3, self.conv_ds_512_4, self.conv_ds_512_5)

        with tf.name_scope('DepthSep_1024'):
            Net._show_weight_and_bias(show_tensor, self.conv_ds_1024_1, self.conv_ds_1024_2)
            Net._show_weight_and_bias(show_grad, self.conv_ds_1024_1, self.conv_ds_1024_2)
