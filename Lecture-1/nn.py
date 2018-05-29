'''
Simple Neural Network example with MNIST dataset
'''
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True) # dataset download

# 변수 선언
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y = Wx + b
y = tf.nn.softmax(tf.matmul(x, W) + b)
# real y
y_ = tf.placeholder("float", [None, 10])

# 학습 모델
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
learn_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for k in range(1000):
    # 100개 data만 추출
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 학습 1단계 진행
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # 성능 평가
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:
                                        mnist.test.labels}))
