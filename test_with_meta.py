import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
test_image = mnist.test.images[:5000]
test_label = mnist.test.labels[:5000]

print()
print("test with .meta: ")

merged = tf.summary.merge_all()
with tf.Session() as sess:
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
    writer = tf.summary.FileWriter('./log/test_with_meta', sess.graph)
    print("accuracy is: ", acc)
writer.close()    