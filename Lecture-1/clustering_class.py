'''
Tensorflow clustering example
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Clustering(object):
    '''
    TF clustering class
    '''
    def __init__(self, points, clusters):
        self.points = points
        self.clusters = clusters
        self.vector = self.set_points()
        self.centroides = self.set_centroides()
        self.assignments = self.cal_assignments()
        self.means = self.cal_means()

    def set_data(self, points, clusters):
        '''set data'''
        self.points = points
        self.clusters = clusters
        self.vector = self.set_points()
        self.centroides = self.set_centroides()
        self.assignments = self.cal_assignments()
        self.means = self.cal_means()

    def set_points(self):
        '''point generator'''
        _points_x = []
        _points_y = []
        for _ in range(self.points):
            if np.random.random() > 0.66666:
                _points_x.append(np.random.normal(0.0, 0.6))
                _points_y.append(np.random.normal(0.0, 0.6))
            elif np.random.random() > 0.33333:
                _points_x.append(np.random.normal(2.0, 0.57))
                _points_y.append(np.random.normal(3.0, 0.6))
            else:
                _points_x.append(np.random.normal(2.0, 0.33))
                _points_y.append(np.random.normal(1.0, 0.2))
        _points = np.array(np.transpose([_points_x, _points_y]))
        _vectors = tf.constant(_points)
        return _vectors

    def set_centroides(self):
        '''set centroides(initial)'''
        _tmp = tf.slice(tf.random_shuffle(self.vector), [0, 0],
                        [self.clusters, -1])
        _centroides = tf.Variable(_tmp)
        return _centroides

    def cal_assignments(self):
        '''calculate assignments'''
        _expanded_vectors = tf.expand_dims(self.vector, 0)
        _expanded_centroides = tf.expand_dims(self.centroides, 1)

        _diff = tf.subtract(_expanded_vectors, _expanded_centroides)
        _distance = tf.reduce_sum(tf.square(_diff), 2)
        _assignments = tf.argmin(_distance, 0)
        return _assignments

    def cal_means(self):
        '''calculate means'''
        _tmp = []
        for _k in range(self.clusters):
            _equal = tf.equal(self.assignments, _k)
            _index = tf.reshape(tf.where(_equal), [1, -1])
            _tensor_idx = tf.gather(self.vector, _index)
            _tmp_means = tf.reduce_mean(_tensor_idx, axis=[1])
            _tmp.append(_tmp_means)
        _means = tf.concat(_tmp, 0)
        return _means


if __name__ == "__main__":
    TEST = Clustering(1000, 3)
    VECTORS = TEST.vector
    CENTROIDES = TEST.centroides
    ASSIGNMENTS = TEST.assignments
    MEANS = TEST.means
    UPDATE_CENTROIDES = tf.assign(CENTROIDES, MEANS)

    INIT_OP = tf.global_variables_initializer()
    SESS = tf.Session()
    SESS.run(INIT_OP)

    for _ in range(100):
        _, centroid_values, assignment_values = SESS.run([UPDATE_CENTROIDES,
                                                          CENTROIDES,
                                                          ASSIGNMENTS])

    TMP_VECTOR = VECTORS.eval(session=SESS)
    POINTS_X = TMP_VECTOR[:, 0]
    POINTS_Y = TMP_VECTOR[:, 1]

    plt.plot(POINTS_X, POINTS_Y, 'ro', label='original data')
    plt.grid()
    plt.legend()
    plt.show()
    # plt.draw()

    print(centroid_values)

    for k, cluster in enumerate(assignment_values):
        if cluster == 0:
            plt.plot(POINTS_X[k], POINTS_Y[k], 'co')
        elif cluster == 1:
            plt.plot(POINTS_X[k], POINTS_Y[k], 'go')
        else:
            plt.plot(POINTS_X[k], POINTS_Y[k], 'bo')

    plt.show()
