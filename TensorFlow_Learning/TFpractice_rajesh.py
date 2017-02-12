#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 14:04:21 2017

@author: rajesh
reference: https://www.youtube.com/watch?v=vq2nnJ4g6N0
"""

import tensorflow as tf
import tensorflowvisu
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

mnist= read_data_sets("data",one_hot=True, reshape= False,validation_size=0)


## Simple model with Softmax
X = tf.placeholder(tf.float32, [None, 28,28,1])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))



# model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X,[-1,784]),W) +b)

# placeholder for correct answers
Y_= tf.placeholder(tf.float32, [None,10])

# Loss Function
cross_entropy = - tf.reduce_mean(Y_ * tf.log(Y))*1000.0 # normalized for batches of 100 images

# % of correct answers found in the batch ( Just for the display, not required in the pipeline)
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.005)
train_step= optimizer.minimize(cross_entropy)

# matplotlib visualization
allweights = tf.reshape(W,[-1])
allbiases = tf.reshape(b,[-1])
I= tensorflowvisu.tf_format_mnist_images(X,Y,Y_) # assembles 10*10 images by default
It = tensorflowvisu.tf_format_mnist_images(X,Y,Y_,1000,lines=25) # 1000 images on 25 lines
datavis= tensorflowvisu.MnistDataVis()

#init
init= tf.global_variables_initializer()

# creating the session
sess= tf.Session()

# initializing the global variables
sess.run(init)

# process will be repeated in a batch of 100 by calling this function
def training_step(i, update_test_data, update_train_data):
    
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    
    train_data= {X: batch_X, Y_: batch_Y}
    test_data= {X: mnist.test.images, Y_:mnist.test.labels}

    # success on the train data?
    if update_train_data:
        a,c,im,w,b = sess.run([accuracy,cross_entropy,I,allweights, allbiases], feed_dict= train_data)
        datavis.append_training_curves_data(i,a,c)
        datavis.append_data_histograms(i,w,b)
        datavis.update_image1(im)
        print (str(i)+ ":accuracy " + str(a) + ":loss " + str(c))
    
    
    # success on the test data?
    if update_test_data:
        a,c,im = sess.run([accuracy,cross_entropy, It], feed_dict=test_data)
        datavis.append_test_curves_data(i,a,c)
        datavis.update_image2(im)
        print (str(i)+ ":****** epoch" + str(i*100//mnist.train.images.shape[0]+1)+ " ******* test accuracy:" + str(a)+ " test loss:" + str(c))
        
      
    # backpropagation training step (this is the main step. Model is getting trained over here..calculate the gradient and update the fraction of it in the W and b)
    sess.run(train_step, feed_dict= train_data)
    
datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start= True)



print ("max test accuracy:" + str(datavis.get_max_test_accuracy()))