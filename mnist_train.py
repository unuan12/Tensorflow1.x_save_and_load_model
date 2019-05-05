# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:35:56 2019

@author: Junyuan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def forward(x):
    # convolution layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # convolution layer 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64]) 
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # full convolution
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # output layer, softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name="output")
    
    return y_conv

def train():
    mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
    
    x = tf.placeholder(tf.float32, [None, 784], name="input")
    y_ = tf.placeholder(tf.float32, [None, 10], name="label")
    
    y_conv = forward(x)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    saver = tf.train.Saver(max_to_keep=3)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(5000):
            batch = mnist.train.next_batch(64)
            if step%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
                print("step %d, training accuracy %g"%(step, train_accuracy))
            if step%1000 == 0:
                saver.save(sess, './model/mnist_model', global_step=step)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    