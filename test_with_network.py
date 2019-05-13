from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mnist_train import forward

mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
test_image = mnist.test.images[:5000]
test_label = mnist.test.labels[:5000]
    
print()
print("test with network: ")

x = tf.placeholder(tf.float32, [None, 784], name="input")
y_ = tf.placeholder(tf.float32, [None, 10], name="label")
pred = forward(x)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, './model/mnist_model-4000')
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(test_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc = sess.run(accuracy, feed_dict={x: test_image, y_: test_label})
    writer = tf.summary.FileWriter('./log/test_with_network', sess.graph)
    print("accuracy is: ", acc)

writer.close()