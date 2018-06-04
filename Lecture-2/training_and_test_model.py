
import tensorflow as tf
import random
import os
import numpy as np
import time
import random
import csv
from random import shuffle

np.random.seed(1117)	
# for reproduct

# parameters
learning_rate = 0.00001
batch_size = 256
nb_epoch = 50
spl = 5
beta2 = 0.9995
feature_dim = 256
layer_width = 2048
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [batch_size, 11*256])
Y = tf.placeholder(tf.float32, [batch_size, feature_dim])

# Input layer
W1 = tf.get_variable("W1", shape=[11*256,layer_width], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([layer_width]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# 1st Hidden layer	
W2 = tf.get_variable("W2", shape=[layer_width,layer_width], initializer=tf.contrib.layers.xavier_initializer())
#W2 = tf.Variable(tf.random_normal([layer_width,layer_width]))
b2 = tf.Variable(tf.random_normal([layer_width]))
L2 = tf.nn.relu(tf.matmul(L1, W2)+ b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# 2nd Hidden layer
W3 = tf.get_variable("W3", shape=[layer_width, layer_width], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([layer_width]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# 3rd Hidden layer
W4 = tf.get_variable("W4", shape=[layer_width,layer_width], initializer=tf.contrib.layers.xavier_initializer())	
b4 = tf.Variable(tf.random_normal([layer_width]))
L4 = tf.nn.relu(tf.matmul(L3, W4)+ b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


# output layer
W5 = tf.get_variable("W5", shape=[layer_width,256], initializer=tf.contrib.layers.xavier_initializer())	
b5 = tf.Variable(tf.random_normal([256]))
hypothesis = tf.matmul(L4, W5) + b5


cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

saver = tf.train.Saver()

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
with tf.Session(config=config) as sess:

	sess.run(tf.global_variables_initializer())
		
	x = input("Do you have saved model : 1 - yes / 2 - no ")
	

	#### TRAINING ####	
	if x == 2:
		
		cost_list = np.array([])
		for a in range(nb_epoch):

			file_1 = open('/home/jhkim17/clean_2000uts.csv', 'r')   ## need modification : clean feature address list
			file_2 = open('/home/jhkim17/noisy_2000uts.csv', 'r')   ## need modification : noise feature address list

			clean_list = file_1.read()
			clean_list = clean_list.split()

			noisy_list = file_2.read()
			noisy_list = noisy_list.split()

			randomlist_original = list(xrange(len(noisy_list)))
			random.shuffle(randomlist_original)
			randomlist = randomlist_original[0:900]
			validlist = randomlist_original[900:1000]
			cost_sum = 0
			avg_cost = 0
			batch_sum = 0

			for list_index, i in enumerate(randomlist):
				start_time = time.time
				target_file = clean_list[i]
				input_file = noisy_list[i]

				noisy_data = np.loadtxt(open(input_file,"rb"), delimiter=',')	
				clean_data = np.loadtxt(open(target_file,"rb"), delimiter=',')

				randomize = np.arange(len(noisy_data))
				np.random.shuffle(randomize)
				
				noisy_data = noisy_data[randomize]
				clean_data = clean_data[randomize]				

				n_batch = int((len(noisy_data)-10)/batch_size)
					
				ci_list = range(spl,len(noisy_data)-spl)
				shuffle(ci_list)
				
				for batch in range(n_batch): 
					ci_batch = ci_list[batch_size*batch:batch_size*(batch+1)]
					batch_con_x = np.array([])
					batch_con_y = np.array([])

					for ci in ci_batch:
						batch_x = noisy_data[ci-spl:ci+spl+1]
						batch_y = clean_data[ci:ci+1]
						(d,w) = batch_x.shape
						batch_x = batch_x.reshape(1,d*w)
							
						batch_con_x = np.append(batch_con_x, batch_x)
						batch_con_y = np.append(batch_con_y, batch_y)
						
					batch_con_x = batch_con_x.reshape(batch_size, d*w)
					batch_con_y = batch_con_y.reshape(batch_size, feature_dim)

					feed_dict = {X: batch_con_x, Y: batch_con_y, keep_prob: 0.5}
					c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
					cost_sum += c
					
				batch_sum += n_batch
			avg_cost = cost_sum/batch_sum
			cost_list = np.append(cost_list, avg_cost)

			print('Epoch:', '%04d' % (a+1), 'cost =', '{:.9f}'.format(avg_cost))
			
			if (a+1)%5 ==0:
				save_path = saver.save(sess, "/home/jhkim17/test_tensorflow/model/epoch_%d" %(a+1))											## need modification : path for deep learning model
				np.savetxt("/home/jhkim17/test_tensorflow/model/DNN3_5times/2nd_12epochs/epoch_#%d.csv" %(a+1), cost_list, delimiter=",")   ## need modification : path for cost
			

			#### Validation ####
			print('%04d th Epoch validation Start' %(a+1))

			cost_sum = 0
			avg_cost = 0
			batch_sum = 0

			for list_index, i in enumerate(validlist):
				target_file = clean_list[i]
				input_file = noisy_list[i]

				noisy_data = np.loadtxt(open(input_file,"rb"), delimiter=',')				
				clean_data = np.loadtxt(open(target_file,"rb"), delimiter=',')				

				n_batch = int((len(noisy_data)-10)/batch_size)
	
				ci_list = range(spl,len(noisy_data)-spl)
				shuffle(ci_list)

				for batch in range(n_batch): 
					ci_batch = ci_list[batch_size*batch:batch_size*(batch+1)]
					batch_con_x = np.array([])
					batch_con_y = np.array([])

					for ci in ci_batch:
						batch_x = noisy_data[ci-spl:ci+spl+1]
						batch_y = clean_data[ci:ci+1]
																																														
						batch_x = batch_x.reshape(1,(2*spl+1)*feature_dim)

						batch_con_x = np.append(batch_con_x, batch_x)
						batch_con_y = np.append(batch_con_y, batch_y)

					batch_con_x = batch_con_x.reshape(batch_size, (2*spl+1)*feature_dim)
					batch_con_y = batch_con_y.reshape(batch_size, feature_dim)
							
					feed_dict = {X: batch_con_x, Y: batch_con_y,keep_prob: 1}
					v_cost, v_hy_val = sess.run([cost, hypothesis], feed_dict=feed_dict)
					cost_sum += v_cost

				print('%d th epoch' %(a+1) ,\
					'%d th valid_file ' %(list_index+1) , 'val_cost = %f' %v_cost )
				batch_sum += n_batch
			
			avg_cost = cost_sum/batch_sum
			v_cost_list = np.append(cost_list, avg_cost)
			
			print('Epoch:', '%04d' % (a+1), 'avg_val_cost =', '{:.9f}'.format(avg_cost))
			if (a+1)%5 == 0:
				np.savetxt("/home/jhkim17/test_tensorflow/model/DNN3_5times/2nd_12epochs/Val_cost_epoch%d.csv" %(a+1), v_cost_list, delimiter=",")	  ## need modification : path for validation cost
			
		save_path = saver.save(sess, "/home/jhkim17/test_tensorflow/model/model")
		np.savetxt("/home/jhkim17/test_tensorflow/model/DNN3_out200_avg_cost_list.csv", cost_list, delimiter=",")


	#### TEST  ####
	else:
		saver.restore(sess, "/home/jhkim17/test_tensorflow/model/model")												## need modification : path for loading model
																				
		test_clean_list = open('/home/jhkim17/clean_cafeteria_SNR0.csv', 'r')											## need modification : path for loading clean test feature
		test_noisy_list = open('/home/jhkim17/noisy_cafeteria_SNR0.csv', 'r')											## need modification : path for noisy loading test feature

		clean_list = test_clean_list.read()
		clean_list = clean_list.split()
		noisy_list = test_noisy_list.read()
		noisy_list = noisy_list.split()
		
		ordered_list = list(xrange(len(noisy_list)))
		ordered_list = ordered_list[0:25]
		
		for list_index, i in enumerate(ordered_list):
						
			start_time = time.time
			target_file = clean_list[i]
			input_file = noisy_list[i]
				
			noisy_data = np.loadtxt(open(input_file,"rb"), delimiter=',')				
			clean_data = np.loadtxt(open(target_file,"rb"), delimiter=',')				

			n_batch = int((len(noisy_data)-10)/batch_size)
				
			ci_list = range(spl,len(noisy_data)-spl)
			output = np.array([])
			output_clean = np.array([])	
			for batch in range(n_batch): 
				ci_batch = ci_list[batch_size*batch:batch_size*(batch+1)]
				batch_con_x = np.array([])
				batch_con_y = np.array([])
																
				for ci in ci_batch:
					batch_x = noisy_data[ci-spl:ci+spl+1]
					batch_y = clean_data[ci:ci+1]
					batch_x = batch_x.reshape(1,(2*spl+1)*feature_dim)

					batch_con_x = np.append(batch_con_x, batch_x)
					batch_con_y = np.append(batch_con_y, batch_y)
						
				batch_con_x = batch_con_x.reshape(batch_size, (2*spl+1)*feature_dim)
				batch_con_y = batch_con_y.reshape(batch_size, feature_dim)
				
				feed_dict = {X: batch_con_x, keep_prob: 1}
				hy_val = sess.run(hypothesis, feed_dict=feed_dict)
				print('hy_val\n')
				print(hy_val)
					
				if batch == 0:
					output = hy_val
					output_clean = batch_con_y
				else:
					output = np.concatenate((output, hy_val),axis=0)
					output_clean = np.concatenate((output_clean, batch_con_y),axis=0)
			np.savetxt('/home/jhkim17/test_tensorflow/testset_cafeteria/DNN_SNR0/output_'+str(list_index+1)+'.csv', output, delimiter=",")		        ## need modification : path for enhanced noisy feature
			np.savetxt('/home/jhkim17/test_tensorflow/testset_cafeteria/DNN_SNR0/output_clean_'+str(list_index+1)+'.csv', output_clean, delimiter=",")  ## need modification : path for clean feature
														
			print('%d th file ' %(list_index+1) , '%d th iteration' %(batch+1) )
																		
																
