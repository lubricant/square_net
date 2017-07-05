
import tensorflow as tf


from model.network import SquareNet


def init_flags():
    flags = tf.app.flags
    flags.DEFINE_integer('image_height', 100, '')
    flags.DEFINE_integer('image_width', 100, '')
    flags.DEFINE_integer('batch_size', 100, '')

    flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
    flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
    flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
    flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')
