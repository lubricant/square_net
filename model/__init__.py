
import tensorflow as tf

import data
from model.network import SquareNet


flags = tf.app.flags
flags.DEFINE_string('exec_mode', data.EM_TEST, 'Application execute mode.')

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('keep_prob', 0.9, 'Keep probability for training dropout.')

flags.DEFINE_integer('label_num', data.NUM_CLASSES, 'Image data size.')
flags.DEFINE_integer('image_size', data.IMG_SIZE, 'Image data size.')
flags.DEFINE_integer('image_channel', data.IMG_CHANNEL, 'Image data size.')
flags.DEFINE_integer('batch_size', 100, 'Batch size of each step.')
flags.DEFINE_integer('epoch_num', 10, 'Number of epochs to run trainer.')
flags.DEFINE_integer("thread_num", 1, 'Number of thread to read data.')
flags.DEFINE_integer("log_interval", 10, 'Number of step between each logging.')
flags.DEFINE_integer("checkpoint_interval", 100, 'Number of step between each checkpoint.')

flags.DEFINE_string('log_dir', data.get_path('tmp/summary'), 'Summaries directory.')
flags.DEFINE_string('checkpoint_dir', data.get_path('tmp/checkpoint'), 'Models directory.')
flags.DEFINE_string('trace_file', data.get_path('tmp/trace.ctf.json'), 'Chrome timeline format file.')

