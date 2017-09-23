import numpy as np
from scipy.optimize import check_grad
import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d_1(x,W):
	return tf.nn.conv2d(x, W, strides = [1,4,4,1], padding = 'VALID')

def conv2d_2(x,W):
	return tf.nn.conv2d(x, W, strides = [1,2,2,1], padding = 'VALID')

def conv2d_3(x,W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')

def max_pool_2x2(x):
	return tf.nn.max_pool_2x2(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


first_conv_net = tf.Graph()

with first_conv_net.as_default():

	image = tf.placeholder(tf.float32,shape = (None, 84*84*4))
	x_image = tf.reshape(image, [-1,84,84,4])
	label = tf.placeholder(tf.float32, shape = (None,10))#not yet known

	#5x5 convolution layer, pool 2x2, depth 1 --> depth 32
	W_conv1 = weight_variable([8,8,4,32])
	b_conv1 = bias_variable([20,20,32])
	h_conv1 = tf.nn.relu(conv2d_1(x_image,W_conv1) + b_conv1)
	#h_pool1 = max_pool_2x2(h_conv1)

	#5x5 convolution layer, pool 2x2, depth 32 --> depth 64
	W_conv2 = weight_variable([4,4,32,64])
	b_conv2 = bias_variable([9,9,64])
	h_conv2 = tf.nn.relu(conv2d_2(h_conv1, W_conv2) + b_conv2)
	#h_pool2 = max_pool_2x2(h_conv2)

	#5x5 convolution layer, pool 2x2, depth 32 --> depth 64
	W_conv3 = weight_variable([3,3,64,64])
	b_conv3 = bias_variable([7,7,64])
	h_conv3 = tf.nn.relu(conv2d_3(h_conv2, W_conv3) + b_conv3)
	#h_pool3 = max_pool_2x2(h_conv2)

	input_dim = 7*7*64
	#Flatten the filtered images in a vector
	h_conv3_flat = tf.reshape(h_conv3, [-1, input_dim])

	keep_prob = tf.placeholder(tf.float32)
	#Fully connected layer of 1024 neurons (activation function: ReLU) with dropout(prob = keep_prob)
	W_fc1 = weight_variable([input_dim,512])
	b_fc1 = bias_variable([512])
	h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	#Fully connected layer into 10 labels (activation function: Softmax)
	W_fc2 = weight_variable([1024, 18])
	b_fc2 = bias_variable([18])
	scores = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #real value
	#y_conv = tf.nn.softmax(scores)

	#Loss function: cross-entropy with softmax loss; numerically stable way
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = scores))

	#Optimization algorithm: ADAM, see https://arxiv.org/abs/1412.6980
	#train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

	#Compute performance
	#correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(label,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

