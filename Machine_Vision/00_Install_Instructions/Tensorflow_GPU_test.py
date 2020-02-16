﻿import tensorflow as tf

print('Using tensorflow version: {} ...'.format(tf.__version__))
print('Visible devices for tensorflow: {} ...'.format(tf.config.list_physical_devices()))

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c_cpu = tf.matmul(a, b)

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c_gpu = tf.matmul(a, b)

print ('Matrix multiplication result using CPU: {}   \
        \nMatrix multiplication result using GPU: {}'.format(c_cpu, c_gpu))