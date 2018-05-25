'''
Tensorflow clustering example
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

NUM_POINTS = 2000
NUM_CLUSTERS = 3
POINTS = tf.zeros([2, NUM_POINTS], dtype=tf.float32)
POINTS_X = []
POINTS_Y = []

for k in range(NUM_POINTS):
    if np.random.random() > 0.66666:
        temp = tf.random_uniform([1], )
        np.random.normal(0.0, 0.55)
        POINTS[0, k] = temp
        POINTS[1, k] = tmpe 
        + np.random.normal(0.0, 0.4)
    elif np.random.random() > 0.33333:
        POINTS_X.append(np.random.normal(2.0, 0.57))
        POINTS_Y.append(np.random.normal(3.0, 0.6))
    else:
        POINTS_X.append(np.random.normal(2.0, 0.33))
        POINTS_Y.append(1 + np.random.normal(0.0, 0.2))

plt.plot(POINTS_X, POINTS_Y, 'ro', label='Original data')
plt.grid()
plt.legend()
plt.show()
