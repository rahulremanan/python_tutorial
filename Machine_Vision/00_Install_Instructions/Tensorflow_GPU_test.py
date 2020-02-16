import tensorflow as tf

print('Using tensorflow version: {} ...'.format(tf.__version__))
print('Visible devices for tensorflow: {} ...'.format(tf.config.list_physical_devices()))

print('Running tensorflow test: 1 ...')
print(tf.reduce_sum(tf.random.normal([1000, 1000])))

print('Running a tensorflow test: 2 ... ')

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c_cpu = tf.matmul(a, b)

c_gpu = ['Failed ...']
try:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c_gpu = tf.matmul(a, b)
except:
    print('Tensorflow test using GPU failed ...')

print ('Matrix multiplication result using CPU: {}   \
        \nMatrix multiplication result using GPU: {}'.format(c_cpu, c_gpu))