#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:42:16 2017

@author: rajesh
"""

import sys
import numpy as np
import tensorflow as tf

from datetime import datetime

device_name = sys.argv[1]
shape = (int(sys.argv[2]), int(sys.argv[2]))

if device_name == 'gpu':
    device_name = '/gpu:0'
else:
    device_name = '/cpu:0'
    
with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0.0, maxval=1.0)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

start_time = datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    result = session.run(sum_operation)
    print(result)
    
print('\n' * 5)
print('shape', shape)
print('device', device_name)
print('time taken:' + str(start_time.now() - start_time))
print('\n' * 5)
