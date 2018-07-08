import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

gamma = 0.99


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1.0/shape[0], shape=shape)
    return tf.Variable(initial)


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


x = tf.placeholder(tf.float32, [None, 4])

W1 = weight_variable([4, 8])
b1 = bias_variable([8])
h1 = tf.nn.relu(tf.matmul(x, W1)+b1)

W2 = weight_variable([8, 4])
b2 = bias_variable([4])
h2 = tf.nn.relu(tf.matmul(h1, W2)+b2)

W3 = weight_variable([4, 2])
b3 = bias_variable([2])
y = tf.nn.softmax(tf.nn.relu(tf.matmul(h2, W3)+b3))

y_ = tf.placeholder(tf.uint8, [None])
yhot = tf.one_hot(y_, depth=2)
rewards = tf.placeholder(tf.float32, [None])

loss = tf.tensordot(rewards, tf.reduce_sum(-yhot * tf.log(y), reduction_indices=[1]), axes=1)

training_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

env = gym.make('CartPole-v0')

ticker_s = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run_num = 0
    for _ in range(2000):
        run_num += 1
        observation = env.reset()
        o_s = [observation]
        a_s = []
        r_s = []
        ticker = 0
        while True:
            if run_num % 50 == 0:
                env.render()
            prob_0 = sess.run(y, feed_dict={x: [observation]})[0][0]
            if np.random.random() < prob_0:
                action = 0
            else:
                action = 1
            a_s += [action]
            observation, reward, done, info = env.step(action)
            ticker += int(not done)
            r_s += [reward]
            if done:
                break
            o_s += [observation]

            print(ticker)
            ticker_s.append(ticker)
            discounted_rewards = discount_rewards(r_s)
            normalized_rewards = (discounted_rewards-np.mean(discounted_rewards))/(np.std(discounted_rewards))
            sess.run(training_step, feed_dict={x: o_s, y_: a_s, rewards: normalized_rewards})

plt.plot(list(range(len(ticker_s))), ticker_s)
plt.show()
