import tensorflow as tf


num_in = 3

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0/shape[0], shape=shape)
    return tf.Variable(initial)

#(a and b) or c = d
fit_in = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
fit_out = [[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[0,1],[0,1]]

x = tf.placeholder(dtype=tf.float32, shape=[None,num_in]) #training input
y_ = tf.placeholder(dtype=tf.float32, shape=[None,2]) #labels

#Input value
x     =   tf.placeholder(tf.float32, [None, num_in])

#Layer 1
W1    =   weight_variable([num_in,8])
b1    =   bias_variable([8])
h1    =   tf.matmul(x,W1)+b1

#Layer 2
W2    =   weight_variable([8,4])
b2    =   bias_variable([4])
h2    =   tf.matmul(h1,W2)+b2

#Layer 3 (output)
W3    =   weight_variable([4,2])
b3    =   bias_variable([2])
y     =   tf.nn.softmax(tf.nn.relu(tf.matmul(h2,W3)+b3))

error = y - y_
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)
predict = tf.argmax(tf.transpose(y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.05)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        _, l = sess.run([train,loss], feed_dict={x:fit_in, y_:fit_out})
        
        if i % 100 == 0:
            print(sess.run(predict, feed_dict={x:fit_in}))
#            print(l)
            
    for elem in fit_out:
        print(elem)
