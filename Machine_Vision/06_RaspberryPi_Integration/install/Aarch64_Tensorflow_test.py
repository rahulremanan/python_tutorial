import subprocess
print((subprocess.check_output("lscpu", shell=True).strip()).decode())
import tensorflow as tf

print('Using tensorflow version: {} ...'.format(tf.__version__))
print('Visible devices for tensorflow: {} ...'.format(tf.config.list_physical_devices()))

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c_cpu = tf.matmul(a, b)

print ('\nMatrix multiplication result using CPU: \n{}'.format(c_cpu))