import sys
import logging

import tensorflow as tf
from tensorflow.python.client import device_lib

import data

'''
    TensorFlow global configuration
'''

flags = tf.app.flags
flags.DEFINE_boolean('is_training', True, 'Application execute mode.')
flags.DEFINE_boolean('exp_decay', True, 'Applying exponential decay to the learning rate.')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.9, 'Base number of exponential decay rate.')
flags.DEFINE_float('keep_prob', 0.6, 'Keep probability for training dropout.')

flags.DEFINE_integer('label_num', data.NUM_CLASSES, 'Number of classes.')
flags.DEFINE_integer('image_size', data.IMG_SIZE, 'Size of image.')
flags.DEFINE_integer('image_channel', data.IMG_CHANNEL, 'Channel of image.')

flags.DEFINE_integer('decay_interval', 3000, 'Number of step between each exponential decay.')
flags.DEFINE_integer("log_interval", 100, 'Number of step between each logging.')
flags.DEFINE_integer("checkpoint_interval", 1000, 'Number of step between each checkpoint.')

flags.DEFINE_string('log_dir', data.TEMP_ROOT + '/tmp/summary', 'Summaries directory.')
flags.DEFINE_string('checkpoint_dir', data.TEMP_ROOT + '/tmp/checkpoint', 'Checkpoint directory.')
flags.DEFINE_string('checkpoint_file', data.TEMP_ROOT + '/tmp/checkpoint/status', 'Checkpoint file prefix.')
flags.DEFINE_string('trace_file', data.TEMP_ROOT + '/tmp/trace.ctf.json', 'Chrome timeline format file.')

if tf.app.flags.FLAGS.is_training:
    flags.DEFINE_string('data_set', data.DS_TRAIN, 'Application execute mode.')
    flags.DEFINE_integer('epoch_num', 3, 'Number of epochs to run trainer.')
    flags.DEFINE_integer("thread_num", 5, 'Number of thread to read data.')
    flags.DEFINE_integer('batch_size', 450, 'Batch size of each step.')
else:
    flags.DEFINE_string('data_set', data.DS_TEST, 'Application execute mode.')
    flags.DEFINE_integer('epoch_num', 1, 'Number of epochs to run trainer.')
    flags.DEFINE_integer("thread_num", 1, 'Number of thread to read data.')
    flags.DEFINE_integer('batch_size', 100, 'Batch size of each step.')

'''
    Custom logging config
'''

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)
COLOR_SEQ, RESET_SEQ = "\033[1;%dm", "\033[0m"

logging.addLevelName(logging.DEBUG, COLOR_SEQ % GREEN + 'DEBUG' + RESET_SEQ)
logging.addLevelName(logging.INFO, COLOR_SEQ % CYAN + 'INFO' + RESET_SEQ )
logging.addLevelName(logging.WARNING, COLOR_SEQ % YELLOW + 'WARN' + RESET_SEQ)
logging.addLevelName(logging.ERROR, COLOR_SEQ % RED + 'ERROR' + RESET_SEQ)
logging.addLevelName(logging.CRITICAL, COLOR_SEQ % MAGENTA + 'CRITICAL' + RESET_SEQ)


'''
    Print config
'''

FLAGS = tf.app.flags.FLAGS
FLAGS._parse_flags()

flags_detail = {bool: [], int: [], float: [], str: []}
[flags_detail[type(v)].append('%- 20s: %- 50s' % (k, str(v))) for k, v in FLAGS.__dict__['__flags'].items()]

device_info = ['%- 20s: %- 10s %+ 20s MB' % (
    x.name, x.device_type, x.memory_limit//1024//1024) for x in device_lib.list_local_devices()]

print('-'*75)
print('|\t' + '\n|\t'.join(flags_detail[bool]))
print('-'*75)
print('|\t' + '\n|\t'.join(flags_detail[str]))
print('-'*75)
print('|\t' + '\n|\t'.join(flags_detail[int]))
print('-'*75)
print('|\t' + '\n|\t'.join(flags_detail[float]))
print('-'*75)
print('|\t' + '\n|\t'.join(device_info))
print('-'*75)
