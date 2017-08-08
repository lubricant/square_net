
import tensorflow as tf
import model.layer as layer


class HCCR_GoogLeNet(object):

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

        self.pool4 = layer.pooling('AvgPool_5x5', [5, 5], 'AVG', stride=3)(self.incp4)
        self.conv4 = layer.convolution('Conv_1x1x128', [1, 1, 128])(self.pool4)

        with layer.default([layer.density], random='gauss:0.01'):
            self.fc = layer.density('FC_1024', 1024)(self.conv4)
            self.dropout = layer.dropout('Dropout')(self.fc)
            self.logits = layer.density('FC_3755', 3755, linear=True)(self.dropout)
            self.keep_prob = self.dropout.vars.keep_prob

        self.loss = layer.loss('Loss')(self.logits, self.labels)

    def __build_summary(self):

        def show_scalar(**args):
            for name in args:
                tf.summary.scalar(name, args[name])

        def show_tensor(**args):
            for name, val in args.items():
                with tf.name_scope(name):
                    mean = tf.reduce_mean(val)
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(val - mean)))
                    tf.summary.scalar('mean', mean)
                    tf.summary.scalar('stddev', stddev)

                    grad, = tf.gradients(self.loss, val)
                    grad_mean = tf.reduce_mean(grad)
                    tf.summary.scalar('grad_mean', grad_mean)

                    tf.summary.histogram('hist', val)
                    tf.summary.histogram('grad', grad)

        def show_weight_and_bias(*args):
            for var in args:
                with tf.name_scope(var.naming):
                    show_tensor(weight=var.vars.weight, bias=var.vars.bias)

        def show_branch_graph(*args):
            for var in args:
                with tf.name_scope(var.naming):
                    for branch in var.ops.graph:
                        if len(branch) == 1:
                            _, branch_node = branch[0]
                            show_weight_and_bias(branch_node)
                        else:
                            branch_name = '-'.join(map(lambda x: x[0], branch))
                            with tf.name_scope(branch_name):
                                show_weight_and_bias(*map(lambda x: x[1], branch))

        with tf.name_scope('Performance'):
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(self.labels, tf.argmax(self.logits, 1)), tf.float32))
            show_scalar(loss=self.loss, accuracy=self.accuracy)

        with tf.name_scope('Convolution'):
            show_weight_and_bias(self.conv1, self.conv2, self.conv3, self.conv4)

        with tf.name_scope('Inception'):
            show_branch_graph(self.incp1, self.incp2, self.incp3, self.incp4)

        with tf.name_scope('FullyConnected'):
            show_weight_and_bias(self.fc, self.logits)
