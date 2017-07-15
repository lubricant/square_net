
import tensorflow as tf
import model.layer as layer


class HCCR_GoogLeNet(object):

    def __init__(self):
        self.__build_network()
        self.__build_summary()

    def __build_network(self):
        assert not tf.get_variable_scope().original_name_scope

        FLAGS = tf.app.flags.FLAGS

        self.images = layer.data('Input', [None, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channel])
        self.labels = layer.data('Label', [None], tf.int64)

        self.conv1 = layer.convolution('Conv_7x7x64', [7, 7, 64], 2)(self.images)
        self.pool1 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(self.conv1)
        self.norm1 = layer.normalization('LocalRespNorm')(self.pool1)

        self.conv2 = layer.convolution('Conv_1x1x64', [1, 1, 64])(self.norm1)
        self.conv3 = layer.convolution('Conv_3x3x192', [3, 3, 192])(self.conv2)
        self.norm2 = layer.normalization('LocalRespNorm')(self.conv3)
        self.pool2 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(self.norm2)

        self.incp1 = layer.inception('Inception_v1',
                                     [('pool_3x3', 32)],
                                     [('conv_1x1', 64)],
                                     [('conv_1x1', 96), ('conv_3x3', 128)],
                                     [('conv_1x1', 16), ('conv_5x5', 32)])(self.pool2)

        self.incp2 = layer.inception('Inception_v1',
                                     [('pool_3x3', 64)],
                                     [('conv_1x1', 128)],
                                     [('conv_1x1', 128), ('conv_3x3', 192)],
                                     [('conv_1x1', 32), ('conv_5x5', 96)])(self.incp1)

        self.pool3 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(self.incp2)

        self.incp3 = layer.inception('Inception_v1',
                                     [('pool_3x3', 64)],
                                     [('conv_1x1', 160)],
                                     [('conv_1x1', 112), ('conv_3x3', 224)],
                                     [('conv_1x1', 24), ('conv_5x5', 64)])(self.pool3)

        self.incp4 = layer.inception('Inception_v1',
                                     [('pool_3x3', 128)],
                                     [('conv_1x1', 256)],
                                     [('conv_1x1', 160), ('conv_3x3', 320)],
                                     [('conv_1x1', 32), ('conv_5x5', 128)])(self.incp3)

        self.pool4 = layer.pooling('MaxPool_5x5', [5, 5], 'MAX', stride=3)(self.incp4)
        self.conv4 = layer.convolution('Conv_1x1x128', [1, 1, 128])(self.pool4)
        self.logits = layer.density('FC_%d' % FLAGS.label_num, FLAGS.label_num, linear=True)(self.conv4)

        self.loss = layer.loss('Loss')(self.logits, self.labels)

        with tf.name_scope('Accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(self.labels, tf.argmax(self.logits, 1)), tf.float32))

    def __build_summary(self):

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.loss)

