import os

import tensorflow as tf

from data import CasiaFile, TFRecordFile


writer = TFRecordFile('test.tfrecords')
for ch, img in CasiaFile('1001-c.gnt'):
    writer.write(ch, img)
writer.close()
