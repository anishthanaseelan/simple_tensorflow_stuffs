import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.load_model('my_model.h5')

model.evaluate(x_test, y_test)

print ( x_test[1] );
print ( y_test[1] );