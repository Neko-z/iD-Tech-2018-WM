import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.5
iterations = 1000

input_data  = tf.placeholder(tf.float32, shape=[None, 784], name="input_data")
output_data = tf.placeholder(tf.float32, shape=[None, 10], name="output_data")

weights = tf.Variable(tf.zeros([784, 10]))
fix = tf.transpose(weights)
biases = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()


model_operation = tf.matmul(input_data, weights) + biases

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=output_data, logits=model_operation), name="cross_entropy"
)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(model_operation, 1), tf.argmax(output_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

w = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(iterations):
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={input_data:batch[0], output_data:batch[1]})
    w = sess.run(fix)

    print(accuracy.eval(feed_dict={input_data: mnist.test.images, output_data: mnist.test.labels}))

for num in w:
    num = num.reshape(28,28)
    plt.gray()
    plt.imshow(num)
    plt.show()



