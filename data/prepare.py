import os

import tensorflow as tf

from data import CasiaFile, TFRecordFile


writer = TFRecordFile('test.tfrecords', TFRecordFile.ch_dict()[1])
for ch, img in CasiaFile('1001-c.gnt'):
    writer.write(ch, img)
writer.close()


for ch, img in TFRecordFile('test.tfrecords'):
    print(ch, img)
    break
