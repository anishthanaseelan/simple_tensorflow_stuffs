#This test script is taken from https://www.tensorflow.org/install/install_windows
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))