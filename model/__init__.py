import sys
import logging

import tensorflow as tf

import data

'''
    TensorFlow global configuration
'''

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


'''
    Custom logging config
'''

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)
COLOR_SEQ, RESET_SEQ = "\033[1;%dm", "\033[0m"

logging.addLevelName(logging.DEBUG, COLOR_SEQ % BLUE + 'DEBUG' + RESET_SEQ)
logging.addLevelName(logging.INFO, COLOR_SEQ % GREEN + 'INFO' + RESET_SEQ )
logging.addLevelName(logging.WARNING, COLOR_SEQ % YELLOW + 'WARN' + RESET_SEQ)
logging.addLevelName(logging.ERROR, COLOR_SEQ % RED + 'ERROR' + RESET_SEQ)
logging.addLevelName(logging.CRITICAL, COLOR_SEQ % MAGENTA + 'CRITICAL' + RESET_SEQ)


'''
    Print config
'''

FLAGS = tf.app.flags.FLAGS
FLAGS._parse_flags()

flags_detail = {int: [], float: [], str: []}
[flags_detail[type(v)].append('%- 20s: %- 50s' % (k, str(v))) for k, v in FLAGS.__dict__['__flags'].items()]

print('-'*75)
print('|\t' + '\n|\t'.join(flags_detail[str]))
print('-'*75)
print('|\t' + '\n|\t'.join(flags_detail[int]))
print('-'*75)
print('|\t' + '\n|\t'.join(flags_detail[float]))
print('-'*75)
