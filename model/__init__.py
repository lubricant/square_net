
import tensorflow as tf

import data
from model.network import SquareNet


flags = tf.app.flags
flags.DEFINE_integer('image_size', 100, 'Image data size.')
flags.DEFINE_integer('batch_size', 100, 'Batch size of each step.')
flags.DEFINE_integer('epoch_num', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer("thread_num", 10, "Number of thread to read data")

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')

flags.DEFINE_string('train_data_dir', data.get_path('tfrecord/train'), 'Training data directory')
flags.DEFINE_string('test_data_dir', data.get_path('tfrecord/test'), 'Test data directory')
flags.DEFINE_string('summaries_dir', data.get_path('tmp/data'), 'Summaries directory')

