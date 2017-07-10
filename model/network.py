
import tensorflow as tf
import model.layer as layer
import data
import model


class SquareNet(object):

    def __init__(self):
        self.__build_network()
        self.__build_summary()

    def __build_network(self):
        assert not tf.get_variable_scope().original_name_scope

        self.image = layer.data('Input', [None, 120, 120, 1])
        self.label = layer.data('Label', [None], tf.int32)

        self.conv1 = layer.convolution('Conv_7x7x64', [7, 7, 64], 2)(self.image)
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

        self.pool4 = layer.pooling('MaxPool_3x3', [5, 5], 'MAX', stride=3)(self.incp4)
        self.conv4 = layer.convolution('Conv_1x1x128', [1, 1, 128])(self.pool4)
        self.fc = layer.density('FC_1024', 1024)(self.conv4)

        self.loss = layer.loss('Softmax')(self.fc, self.label)
        self.loss_sum = tf.reduce_sum(self.loss)
        self.loss_mean = tf.reduce_mean(self.loss)

    def __build_summary(self):

        tf.summary.scalar('loss_sum', self.loss_sum)
        tf.summary.scalar('loss_mean', self.loss_mean)
        self.summary = tf.summary.merge_all()

    # def restore_network(self, sess, path):
    #     saver = tf.train.Saver()
    #     saver.restore(sess, path)
    #
    # def save_network(self, sess, path):
    #     saver = tf.train.Saver()
    #     saver.save(sess, path)
    #
    # def write_summary(self, sess, path):
    #     writer = tf.summary.FileWriter(path)
    #     writer.add_summary(self.summary)
