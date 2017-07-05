
import tensorflow as tf
import model.layer as layer
import data
import model


class SquareNet(object):

    network = object()

    def __init__(self):
        self.__init_flags()
        self.__build_network()
        self.__build_summary()

    def __init_flags(self):
        flags = tf.app.flags
        flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
        flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
        flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
        flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
        flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
        flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')

    def __build_network(self):
        assert not tf.get_variable_scope().original_name_scope

        net = dict()

        image = net['image'] = layer.data('Input', [None, 120, 120, 1])
        label = net['label'] = layer.data('Label', [None], tf.int32)

        conv1 = net['conv1'] = layer.convolution('Conv_7x7x64', [7, 7, 64], 2)(image)
        pool1 = net['pool1'] = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(conv1)
        norm1 = net['norm1'] = layer.normalization('LocalRespNorm')(pool1)

        conv2 = net['conv2'] = layer.convolution('Conv_1x1x64', [1, 1, 64])(norm1)
        conv3 = net['conv3'] = layer.convolution('Conv_3x3x192', [3, 3, 192])(conv2)
        norm2 = net['norm2'] = layer.normalization('LocalRespNorm')(conv3)
        pool2 = net['pool2'] = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(norm2)

        incp1 = net['incp1'] = layer.inception('Inception_v1',
                                               [('pool_3x3', 32)],
                                               [('conv_1x1', 64)],
                                               [('conv_1x1', 96), ('conv_3x3', 128)],
                                               [('conv_1x1', 16), ('conv_5x5', 32)])(pool2)

        incp2 = net['incp2'] = layer.inception('Inception_v1',
                                               [('pool_3x3', 64)],
                                               [('conv_1x1', 128)],
                                               [('conv_1x1', 128), ('conv_3x3', 192)],
                                               [('conv_1x1', 32), ('conv_5x5', 96)])(incp1)

        pool3 = net['pool3'] = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(incp2)

        incp3 = net['incp3'] = layer.inception('Inception_v1',
                                               [('pool_3x3', 64)],
                                               [('conv_1x1', 160)],
                                               [('conv_1x1', 112), ('conv_3x3', 224)],
                                               [('conv_1x1', 24), ('conv_5x5', 64)])(pool3)

        incp4 = net['incp4'] = layer.inception('Inception_v1',
                                               [('pool_3x3', 128)],
                                               [('conv_1x1', 256)],
                                               [('conv_1x1', 160), ('conv_3x3', 320)],
                                               [('conv_1x1', 32), ('conv_5x5', 128)])(incp3)

        pool4 = net['pool4'] = layer.pooling('MaxPool_3x3', [5, 5], 'MAX', stride=3)(incp4)
        conv4 = net['conv4'] = layer.convolution('Conv_1x1x128', [1, 1, 128])(pool4)
        fc = net['fc'] = layer.density('FC_1024', 1024)(conv4)

        loss = net['loss'] = layer.loss('Softmax')(fc, label)
        net['loss_sum'] = tf.reduce_sum(loss)
        net['loss_mean'] = tf.reduce_mean(loss)

        [setattr(self.network, attr, net[attr]) for attr in net]

    def __build_summary(self):

        tf.summary.scalar('loss_sum', self.network.loss_sum)
        tf.summary.scalar('loss_mean', self.network.loss_mean)

        setattr(self.network, 'summary', tf.summary.merge_all())

    def restore_network(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path)

    def save_network(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, path)

    def write_summary(self, sess, path):
        writer = tf.summary.FileWriter(path)
        writer.add_summary(self.network.summary)


model.init_flags()

image_batch, label_batch = data.tf_queue()
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # while not coord.should_stop():
        while True:
            images, labels = sess.run([image_batch, label_batch])
            print(images)
            print(labels)
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)


