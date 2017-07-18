
import tensorflow as tf
import model.layer as layer


class HCCR_GoogLeNet(object):

    def __init__(self):
        with tf.name_scope('HCCR-GoogLeNet'):
            self.__build_network()
        self.__build_summary()

    def __build_network(self):
        assert not tf.get_variable_scope().original_name_scope

        FLAGS = tf.app.flags.FLAGS

        self.images = layer.data('Input', [None, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channel])
        self.labels = layer.data('Label', [None], tf.int64)

        self.conv1 = layer.convolution('Conv_7x7x64', [7, 7, 64], 2, random='gauss:0.015')(self.images)
        self.pool1 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(self.conv1)
        self.norm1 = layer.normalization('LocalRespNorm', alpha=0.0001, beta=0.75)(self.pool1)

        self.conv2 = layer.convolution('Conv_1x1x64', [1, 1, 64])(self.norm1)
        self.conv3 = layer.convolution('Conv_3x3x192', [3, 3, 192], padding='SAME')(self.conv2)
        self.norm2 = layer.normalization('LocalRespNorm', alpha=0.0001, beta=0.75)(self.conv3)
        self.pool2 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(self.norm2)

        self.incp1 = layer.inception('Inception_v1_1',
                                     [('pool_3x3', 32)],
                                     [('conv_1x1', 64)],
                                     [('conv_1x1', 96), ('conv_3x3', 128, {'random': 'gauss:0.04', 'padding': 'SAME'})],
                                     [('conv_1x1', 16), ('conv_5x5', 32, {'random': 'gauss:0.08', 'padding': 'SAME'})])(
                                     self.pool2)

        self.incp2 = layer.inception('Inception_v1_2',
                                     [('pool_3x3', 64)],
                                     [('conv_1x1', 128)],
                                     [('conv_1x1', 128), ('conv_3x3', 192, {'random': 'gauss:0.04', 'padding': 'SAME'})],
                                     [('conv_1x1', 32), ('conv_5x5', 96, {'random': 'gauss:0.08', 'padding': 'SAME'})])(
                                    self.incp1)

        self.pool3 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(self.incp2)

        self.incp3 = layer.inception('Inception_v1_3',
                                     [('pool_3x3', 64)],
                                     [('conv_1x1', 160)],
                                     [('conv_1x1', 112), ('conv_3x3', 224, {'random': 'gauss:0.04', 'padding': 'SAME'})],
                                     [('conv_1x1', 24), ('conv_5x5', 64, {'random': 'gauss:0.08', 'padding': 'SAME'})])(
                                     self.pool3)

        self.incp4 = layer.inception('Inception_v1_4',
                                     [('pool_3x3', 128)],
                                     [('conv_1x1', 256)],
                                     [('conv_1x1', 160), ('conv_3x3', 320, {'random': 'gauss:0.04', 'padding': 'SAME'})],
                                     [('conv_1x1', 32), ('conv_5x5', 128, {'random': 'gauss:0.08', 'padding': 'SAME'})])(
                                     self.incp3)

        self.pool4 = layer.pooling('AvgPool_5x5', [5, 5], 'AVG', stride=3)(self.incp4)
        self.conv4 = layer.convolution('Conv_1x1x128', [1, 1, 128])(self.pool4)

        self.fc = layer.density('FC_1024', 1024)(self.conv4)
        self.dropout = layer.dropout('Dropout_FC_1024')(self.fc)
        self.keep_prob = self.dropout.keep_prob

        self.logits = layer.density('FC_%d' % FLAGS.label_num, FLAGS.label_num, linear=True)(self.dropout)

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
                tf.summary.histogram(name, val)

        def show_weight_and_bias(*args):
            for var in args:
                with tf.name_scope(var.layer_name):
                    show_tensor(weight=var.weight, bias=var.bias)

        def show_branch_graph(*args):
            for var in args:
                with tf.name_scope(var.layer_name):
                    for branch in var.branch_graph:
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

        with tf.name_scope('ConvolutionLayer'):
            show_weight_and_bias(self.conv1, self.conv2, self.conv3, self.conv4)

        with tf.name_scope('InceptionLayer'):
            show_branch_graph(self.incp1, self.incp2, self.incp3, self.incp4)

        with tf.name_scope('FullyConnectedLayer'):
            show_weight_and_bias(self.fc, self.logits)
