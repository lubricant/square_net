
import tensorflow as tf

import data
from model.network import SquareNet


flags = tf.app.flags
flags.DEFINE_integer('image_size', data.IMAGE_SIZE, 'Image data size.')
flags.DEFINE_integer('batch_size', 100, 'Batch size of each step.')
flags.DEFINE_integer('epoch_num', 10, 'Number of epochs to run trainer.')
flags.DEFINE_integer("thread_num", 1, 'Number of thread to read data.')
flags.DEFINE_integer("log_interval", 1100, 'Number of step between each logging.')

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')

flags.DEFINE_string('exec_mode', data.EM_TRAINING, 'Application execute mode.')
flags.DEFINE_string('log_dir', data.get_path('tmp/log'), 'Models directory.')
flags.DEFINE_string('model_dir', data.get_path('tmp/model'), 'Models directory.')
flags.DEFINE_string('summary_dir', data.get_path('tmp/summary'), 'Summaries directory.')

