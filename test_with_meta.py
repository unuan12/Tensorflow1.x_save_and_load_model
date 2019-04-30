# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:24:37 2019

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
test_image = mnist.test.images[:5000]
test_label = mnist.test.labels[:5000]

print()
print("test with .meta: ")

with tf.Session() as sess:
    #https://blog.csdn.net/liuxiao214/article/details/79048136
    # load the meta graph and weights
    saver = tf.train.import_meta_graph('./model/mnist_model-4000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./model/"))
    
    # get weights
    graph = tf.get_default_graph()
    
    input_x = graph.get_operation_by_name("input").outputs[0]
    feed_dict = {"input:0":test_image, "label:0":test_label}
    pred = graph.get_tensor_by_name("output:0")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(test_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc = sess.run(accuracy, feed_dict=feed_dict)
    print("accuracy is: ", acc)