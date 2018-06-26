# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time

##########################################
'''  "Joint Optimization of Masks and Deep Recurrent Neural Networks ''' 
'''  for Monaural Source Separation" (by Soo Hyun Bae) '''
''' test '''
##########################################

#########################
''' Parameters setting'''
#########################
input_dim = 257
n_layer = 3
hidden_size = 500
keep_prob = 0.7
learning_rate = 0.0001
epoch = 50
batch_size = 50
gamma = 0.05
mixed_list = '../feature/mixed_test_list.txt'
model_save_path = './model/model'


#################
''' Functions '''
#################
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [[i*batch_size, min(size, (i+1)*batch_size)] for i in range(0, nb_batch)]

##########################
''' Dataset processing '''
##########################
file_mixed = open(mixed_list, 'r')
mixed_list = file_mixed.read()
mixed_list = mixed_list.split()


########################################
''' Model structure for rnn training '''
########################################
# placeholder
x_mixed = tf.placeholder(tf.float32, shape=[None, None, input_dim]) # [batch_size,max_time,input_dim]
y_src1 = tf.placeholder(tf.float32, shape=[None, None, input_dim])
y_src2 = tf.placeholder(tf.float32, shape=[None, None, input_dim])
sequence_length = tf.placeholder(tf.int32, shape=[None])

# RNN layers
layer1 = tf.nn.rnn_cell.GRUCell(hidden_size)
layer1 = tf.nn.rnn_cell.DropoutWrapper(layer1, output_keep_prob=keep_prob)
layer2 = tf.nn.rnn_cell.GRUCell(hidden_size)
layer2 = tf.nn.rnn_cell.DropoutWrapper(layer2, output_keep_prob=keep_prob)
rnn_layer = tf.nn.rnn_cell.MultiRNNCell([layer1, layer2])
output_rnn, rnn_state = tf.nn.dynamic_rnn(rnn_layer, x_mixed, sequence_length=sequence_length,dtype=tf.float32)

# dense layer
y_hat_src1 = tf.layers.dense(inputs=output_rnn, units=input_dim, activation=tf.nn.relu)
y_hat_src2 = tf.layers.dense(inputs=output_rnn, units=input_dim, activation=tf.nn.relu)

# time-freq masking layer
y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * x_mixed
y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * x_mixed

# loss
loss_mse = tf.reduce_mean(tf.square(y_src1 - y_tilde_src1) + tf.square(y_src2 - y_tilde_src2))
loss_disc = -gamma*tf.reduce_mean(tf.square(y_src1 - y_tilde_src2) + tf.square(y_src2 - y_tilde_src1))
loss_total = 0.5*(loss_mse + loss_disc)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_total)

#################
''' Train rnn '''
#################
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
gpu_options = tf.GPUOptions(allow_growth =True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    recent_ckpt_job_path = tf.train.latest_checkpoint("model")
    saver.restore(sess, recent_ckpt_job_path)
    print('========used model : ', recent_ckpt_job_path)
    
    for i in range(len(mixed_list)):
        file_name = mixed_list[i]
        n_frame = os.path.getsize(file_name)
        n_frame = int(n_frame / 4 / input_dim)  
        data = np.fromfile(file_name, dtype = 'float32')
        y_test_in = data.reshape(1, n_frame, input_dim)
        pred_src1, pred_src2 = sess.run([y_tilde_src1, y_tilde_src2],
                                        feed_dict = { x_mixed: y_test_in, sequence_length: [n_frame]})
        
        f_name = file_name[69:]
        pred_src1 = np.array(pred_src1)
        output_save_path = './output1/' + f_name
        pred_src1.astype('float32').tofile(output_save_path)
        
        pred_src2 = np.array(pred_src2)
        output_save_path = './output2/' + f_name
        pred_src2.astype('float32').tofile(output_save_path)        
        
    print('=====test complete===============') 
    