
import tensorflow as tf


from model.network import SquareNet


def init_flags():
    flags = tf.app.flags
    flags.DEFINE_integer('image_height', 100, '')
    flags.DEFINE_integer('image_width', 100, '')
    flags.DEFINE_integer('batch_size', 100, '')
