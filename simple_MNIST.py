import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets ( "MNIST_data/", one_hot = True) 

x = tf.placeholder ( tf.float32 , shape=[None, 784])

y_ = tf.placeholder ( tf.float32 , shape=[None , 10])

# Initialize Weight and Bias
W = tf.Variable ( tf.zeros([784, 10]) )
b = tf.Variable ( tf.zeros([10]) )

#Our basic Model equation
#Be aware of Ship mismatch that could be caused by order of multiplication elements in Matric multiplication
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels=y_ , logits=y))

training_steps = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range (1000):
    batch_xs , batch_ys = mnist.train.next_batch(100)

    sess.run(training_steps , feed_dict={x:batch_xs , y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))

test_accuracy = sess.run(accuracy , feed_dict={x: mnist.test.images , y_: mnist.test.labels})

print ( "Test Accuracy : {0}%".format(test_accuracy * 100))

sess.close()





