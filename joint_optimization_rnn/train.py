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
''' train '''
##########################################

#########################
''' Parameters setting'''
#########################
input_dim = 257
hidden_size = 500
keep_prob = 0.7
learning_rate = 0.0001
epoch = 50
batch_size = 50
gamma = 0.05
mixed_list = '../feature/mixed_train_list.txt'
clean_list = '../feature/clean_train_list.txt'
noise_list = '../feature/noise_train_list.txt'
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

file_clean = open(clean_list, 'r')
clean_list = file_clean.read()
clean_list = clean_list.split()

file_noise = open(noise_list, 'r')
noise_list = file_noise.read()
noise_list = noise_list.split()

randomlist = list(range(len(mixed_list)))

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
    
    batches = make_batches(len(mixed_list), batch_size)
    for step in range(1, epoch+1):
        start = time.time()
        np.random.shuffle( randomlist )
        loss_epoch = 0
        
        for i, (batch_start, batch_end) in enumerate(batches):
            sequence_len = np.zeros(batch_end-batch_start)
            for j in range(batch_end-batch_start):
                file_name = mixed_list[randomlist[batch_start+j]]
                n_frame = os.path.getsize(file_name)
                n_frame = int(n_frame / 4 / input_dim)
                sequence_len[j] = n_frame
            max_time = int(max(sequence_len))
            batch_tensor_x_mixed = np.zeros((batch_end-batch_start, max_time ,input_dim))
            batch_tensor_y_src1 = np.zeros((batch_end-batch_start, max_time ,input_dim))
            batch_tensor_y_src2 = np.zeros((batch_end-batch_start, max_time ,input_dim))
            for j in range(batch_end-batch_start):
                file_name = mixed_list[randomlist[batch_start+j]]
                n_frame = os.path.getsize(file_name)
                n_frame = int(n_frame / 4 / input_dim)
                data = np.fromfile(file_name, dtype = 'float32')
                data = data.reshape(n_frame, input_dim)
                pad_len = max_time - n_frame
                pad_width = ((0, pad_len), (0, 0))
                padded_src = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
                batch_tensor_x_mixed[j,:,:] = padded_src
                
                file_name = clean_list[randomlist[batch_start+j]]
                data = np.fromfile(file_name, dtype = 'float32')
                data = data.reshape(n_frame, input_dim)
                padded_src = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
                batch_tensor_y_src1[j,:,:] = padded_src
                
                file_name = noise_list[randomlist[batch_start+j]]
                data = np.fromfile(file_name, dtype = 'float32')
                data = data.reshape(n_frame, input_dim)
                padded_src = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
                batch_tensor_y_src2[j,:,:] = padded_src                
            
            _, cur_loss = sess.run([optimizer, loss_total],
                                   feed_dict={x_mixed: batch_tensor_x_mixed,
                                              y_src1: batch_tensor_y_src1,
                                              y_src2: batch_tensor_y_src2,
                                              sequence_length: sequence_len})
            loss_epoch += cur_loss
            
        saver.save(sess, model_save_path, step)
        end = time.time() - start
        
        print("epoch %d -  loss_total: %3.5f, exe_time: %d" %(step, loss_epoch, end))