#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:11:17 2017

@author: rajesh
reference: https://www.youtube.com/watch?v=vq2nnJ4g6N0
"""

import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

mnist= read_data_sets("data",one_hot=True, reshape= False,validation_size=0)

# input X: 28* 28 grayscale images
X = tf.placeholder(tf.float32, [None, 28,28,1])
# correct answers will go here
Y_= tf.placeholder(tf.float32, [None,10])
#variable learning rate
lr = tf.placeholder(tf.float32)
# probability of keeping a node during dropout =1.0 at test time(no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# five layers and their number of neurons ( last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
O = 30

# weights initialized with small random values between -0.2 and +0.2
# When using with ReLu, make sure biases are initialized with small "positive" values for example 0.1 = tf.ones([k])/10

W1= tf.Variable(tf.truncated_normal([784,L], stddev=0.1))
B1= tf.Variable(tf.ones([L])/10)
W2= tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
B2= tf.Variable(tf.ones([M])/10)
W3= tf.Variable(tf.truncated_normal([M,N],stddev=0.1))
B3= tf.Variable(tf.ones([N])/10)
W4= tf.Variable(tf.truncated_normal([N,O],stddev=0.1))
B4= tf.Variable(tf.ones([O])/10)
W5= tf.Variable(tf.truncated_normal([O,10],stddev=0.1))
B5= tf.Variable(tf.zeros([10]))


# the model
Y1= tf.nn.relu(tf.matmul(tf.reshape(X,[-1,784]),W1)+B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2= tf.nn.relu(tf.matmul(Y1d,W2)+B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3= tf.nn.relu(tf.matmul(Y2d,W3)+B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4= tf.nn.relu(tf.matmul(Y3d,W4)+B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits= tf.matmul(Y4d,W5)+B5
Y= tf.nn.softmax(Ylogits)

# tensorflow provides the softmax_cross_entropy_with_logits function to avoid numerical instability
cross_entropy= tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy= tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model between 0 (worst) and 1(best)
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer
train_step= tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# matplotlib visualization
allweights = tf.concat(0,[tf.reshape(W1,[-1]),tf.reshape(W2,[-1]) ,tf.reshape(W3,[-1]),tf.reshape(W4,[-1]),tf.reshape(W5,[-1])])
allbiases  = tf.concat(0,[tf.reshape(B1,[-1]),tf.reshape(B2,[-1]) ,tf.reshape(B3,[-1]),tf.reshape(B4,[-1]),tf.reshape(B5,[-1])])
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
    
    
    # learning rate decay
    max_learning_rate=0.003
    min_learning_rate=0.0001
    decay_speed= 2000.0 # 0.003-0.0001-2000 => 0.9826 done in 5000 iterations
    learning_rate= min_learning_rate+(max_learning_rate -min_learning_rate)*math.exp(-i/decay_speed)
    
    
    #train_data= {X: batch_X, Y_: batch_Y, lr:learning_rate}
    #test_data= {X: mnist.test.images, Y_:mnist.test.labels}


    # success on the train data?
    if update_train_data:
        a,c,im,w,b = sess.run([accuracy,cross_entropy,I,allweights, allbiases], {X: batch_X, Y_: batch_Y, pkeep:0.75})
        datavis.append_training_curves_data(i,a,c)
        datavis.append_data_histograms(i,w,b)
        datavis.update_image1(im)
        print (str(i)+ ":accuracy " + str(a) + ":loss " + str(c) + "(lr:" + str(learning_rate) + ")")
    
    
    # success on the test data?
    if update_test_data:
        a,c,im = sess.run([accuracy,cross_entropy, It], {X: mnist.test.images, Y_:mnist.test.labels, pkeep:1.0})
        datavis.append_test_curves_data(i,a,c)
        datavis.update_image2(im)
        print (str(i)+ ":****** epoch" + str(i*100//mnist.train.images.shape[0]+1)+ " ******* test accuracy:" + str(a)+ " test loss:" + str(c))
        
        
      
    # backpropagation training step (this is the main step. Model is getting trained over here..calculate the gradient and update the fraction of it in the W and b)
    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep:0.75, lr:learning_rate})
    
datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start= True)



print ("max test accuracy:" + str(datavis.get_max_test_accuracy()))
