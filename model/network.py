
import tensorflow as tf
import model.layer as layer


image = layer.data('Input', [None, 120, 120, 1])
label = layer.data('Label', [None], tf.int32)

conv1 = layer.convolution('Conv_7x7x64', [7, 7, 64], 2)(image)
pool1 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(conv1)
norm1 = layer.normalization('LocalRespNorm')(pool1)

conv2 = layer.convolution('Conv_1x1x64', [1, 1, 64])(norm1)
conv3 = layer.convolution('Conv_3x3x192', [3, 3, 192])(conv2)
norm2 = layer.normalization('LocalRespNorm')(conv3)
pool2 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(norm2)

ince1 = layer.inception('Inception_v1',
                        [('pool_3x3', 32)],
                        [('conv_1x1', 64)],
                        [('conv_1x1', 96), ('conv_3x3', 128)],
                        [('conv_1x1', 16), ('conv_5x5', 32)])(pool2)

ince2 = layer.inception('Inception_v1',
                        [('pool_3x3', 64)],
                        [('conv_1x1', 128)],
                        [('conv_1x1', 128), ('conv_3x3', 192)],
                        [('conv_1x1', 32), ('conv_5x5', 96)])(ince1)

pool3 = layer.pooling('MaxPool_3x3', [3, 3], 'MAX', stride=2)(ince2)

ince3 = layer.inception('Inception_v1',
                        [('pool_3x3', 64)],
                        [('conv_1x1', 160)],
                        [('conv_1x1', 112), ('conv_3x3', 224)],
                        [('conv_1x1', 24), ('conv_5x5', 64)])(ince2)

ince4 = layer.inception('Inception_v1',
                        [('pool_3x3', 128)],
                        [('conv_1x1', 256)],
                        [('conv_1x1', 160), ('conv_3x3', 320)],
                        [('conv_1x1', 32), ('conv_5x5', 128)])(ince3)

pool4 = layer.pooling('MaxPool_3x3', [5, 5], 'MAX', stride=3)(ince4)


conv4 = layer.convolution('Conv_1x1x128', [1, 1, 128])(pool4)
fc = layer.density('FC_1024', 1024)(conv4)

loss = layer.loss('Softmax')(fc, label)



