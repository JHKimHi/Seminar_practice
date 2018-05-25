'''
Tensorflow example: linear regression
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

NUM_POINTS = 1000
X = []
Y = []

for k in range(NUM_POINTS):
    temp = np.random.normal(0.0, 0.55)
    X.append(temp)
    Y.append(temp * 0.1 + 0.3 + np.random.normal(0.0, 0.03))

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
Y_MK = W * X + B

plt.plot(X, Y, 'ro', label='Original data')
plt.grid()
plt.legend()
plt.draw()

loss = tf.reduce_mean(tf.square(Y_MK - Y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(8):
    sess.run(train)
    print(step, sess.run(W), sess.run(B))
    print(step, sess.run(loss))

    # Graphic display
    plt.plot(X, sess.run(W) * X + sess.run(B))
    plt.xlabel('X')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('Y')
    plt.legend()
    plt.draw()
    plt.pause(0.5)

time.sleep(10)
