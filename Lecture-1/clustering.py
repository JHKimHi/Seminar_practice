'''
Tensorflow clustering example
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1000
num_clusters = 3
points_x = []
points_y = []

# 1000개의 랜덤 포인트를 3개의 클러스터로 생성
for k in range(num_points):
    if np.random.random() > 0.66666:
        points_x.append(np.random.normal(0.0, 0.6))
        points_y.append(np.random.normal(0.0, 0.6))
    elif np.random.random() > 0.33333:
        points_x.append(np.random.normal(2.0, 0.57))
        points_y.append(np.random.normal(3.0, 0.6))
    else:
        points_x.append(np.random.normal(2.0, 0.33))
        points_y.append(np.random.normal(1.0, 0.2))

# 1000개의 포인트를 numpy array로 생성
points = np.array(np.transpose([points_x, points_y]))

# numpy array를 tensor로 변환
vectors = tf.constant(points)
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0],
                                  [num_clusters, -1])) # 중심점 초기화

# 텐서 차원 확장
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

# 거리 계산
diff = tf.subtract(expanded_vectors, expanded_centroides)
distance = tf.reduce_sum(tf.square(diff), 2)
assignments = tf.argmin(distance, 0)

# 중심점 업데이트
means = tf.concat([tf.reduce_mean(tf.gather(
    vectors, tf.reshape(tf.where(tf.equal(
        assignments, c)), [1, -1])), axis=[1])
                   for c in range(num_clusters)], 0)


update_centroides = tf.assign(centroides, means)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroides,
                                                      centroides, assignments])

plt.plot(points_x, points_y, 'ro', label='original data')
plt.grid()
plt.legend()
plt.show()
# plt.draw()
plt.pause(0.01)

print(centroid_values)

for k, cluster in enumerate(assignment_values):
    if cluster == 0:
        plt.plot(points_x[k], points_y[k], 'co')
    elif cluster == 1:
        plt.plot(points_x[k], points_y[k], 'go')
    else:
        plt.plot(points_x[k], points_y[k], 'bo')

plt.show()
