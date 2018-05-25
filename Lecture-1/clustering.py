'''
Tensorflow clustering example
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

NUM_POINTS = 1000
NUM_CLUSTERS = 3
POINTS_X = []
POINTS_Y = []

for k in range(NUM_POINTS):
    if np.random.random() > 0.66666:
        POINTS_X.append(np.random.normal(0.0, 0.6))
        POINTS_Y.append(np.random.normal(0.0, 0.6))
    elif np.random.random() > 0.33333:
        POINTS_X.append(np.random.normal(2.0, 0.57))
        POINTS_Y.append(np.random.normal(3.0, 0.6))
    else:
        POINTS_X.append(np.random.normal(2.0, 0.33))
        POINTS_Y.append(1 + np.random.normal(0.0, 0.2))

# POINTS = np.array([np.transpose(POINTS_X), np.transpose(POINTS_Y)])
POINTS = np.array(np.transpose([POINTS_X, POINTS_Y]))

VECTORS = tf.constant(POINTS)
CENTROIDES = tf.Variable(tf.slice(tf.random_shuffle(VECTORS), [0, 0],
                                  [NUM_CLUSTERS, -1]))
EXPANDED_VECTORS = tf.expand_dims(VECTORS, 0)
EXPANDED_CENTROIDES = tf.expand_dims(CENTROIDES, 1)

ASSIGNMENTS = tf.argmin(tf.reduce_sum(tf.square(
    tf.subtract(EXPANDED_VECTORS, EXPANDED_CENTROIDES)), 2), 0)
MEANS = tf.concat([tf.reduce_mean(tf.gather(
    VECTORS, tf.reshape(tf.where(tf.equal(
        ASSIGNMENTS, c)), [1, -1])), axis=[1])
                   for c in range(NUM_CLUSTERS)], 0)

UPDATE_CENTROIDES = tf.assign(CENTROIDES, MEANS)

INIT_OP = tf.initialize_all_variables()

SESS = tf.Session()
SESS.run(INIT_OP)

for step in range(100):
    _, centroid_values, assignment_values = SESS.run([UPDATE_CENTROIDES,
                                                      CENTROIDES, ASSIGNMENTS])

plt.plot(POINTS_X, POINTS_Y, 'ro', label='Original data')
plt.grid()
plt.legend()
plt.draw()
plt.pause(0.01)

print(centroid_values)

for k, cluster in enumerate(assignment_values):
    if cluster == 0:
        plt.plot(POINTS_X[k], POINTS_Y[k], 'co')
        plt.draw()
        plt.pause(0.0001)
    elif cluster == 1:
        plt.plot(POINTS_X[k], POINTS_Y[k], 'go')
        plt.draw()
    else:
        plt.plot(POINTS_X[k], POINTS_Y[k], 'bo')
        plt.draw()
