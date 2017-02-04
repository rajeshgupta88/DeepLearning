# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf
import argparse


x = tf.placeholder(tf.float32,[None,784])


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# y is the predicted probability distribution
y = tf.nn.softmax(tf.matmul(x,W)+b)

# y_ is the true distribution ( one-hot vector with the digit labels)
y_ = tf.placeholder(tf.float32, [None, 10])


# implementing the cross entropy function ( -sum(y_log(y)) which serves as a cost function
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Created the session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# let's train the model- we will run the training step 1000 times
for i in range (100000):
    batch_xs, batch_ys= mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict= {x: batch_xs, y_: batch_ys})
    
# Using small batches of random data is called stochastic training ( in this case, stochastic gradient descent)


# Evaluating the model now
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Finally we ask for our accuracy on our test data
print (sess.run(accuracy, feed_dict= {x: mnist.test.images, y_: mnist.test.labels}))

#batch_size=10000
#print(sess.run(accuracy, feed_dict={x: mnist.test.images[0:batch_size], y_: mnist.test.labels[0:batch_size]}))


  






