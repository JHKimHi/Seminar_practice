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

# 랜덤 1000포인트 생성
for k in range(NUM_POINTS):
    temp = np.random.normal(0.0, 0.55)
    X.append(temp)
    Y.append(temp * 0.1 + 0.3 + np.random.normal(0.0, 0.03))

# weigth와 bias를 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
Y_MK = W * X + B

# 초기 점의 위치를 빨간색으로 표시
plt.plot(X, Y, 'ro', label='Original data')
plt.grid()
plt.legend()
plt.draw()

# TF 모델 구조
LOSS = tf.reduce_mean(tf.square(Y_MK - Y))
OPTIMIZER = tf.train.GradientDescentOptimizer(0.5)
TRAIN = OPTIMIZER.minimize(LOSS)

INIT = tf.global_variables_initializer()

# TF 세션을 실행
SESS = tf.Session()
SESS.run(INIT)

# 8번의 스텝 진행
for step in range(8):
    SESS.run(TRAIN)
    print(step, SESS.run(W), SESS.run(B))
    print(step, SESS.run(LOSS))

    # Graphic display
    plt.plot(X, SESS.run(W) * X + SESS.run(B))
    plt.xlabel('X')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('Y')
    plt.legend()
    plt.draw()
    plt.pause(0.5)

input("엔터를 눌러 종료")
