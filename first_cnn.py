#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from scipy.optimize import check_grad

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json

import tensorflow as tf



GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
OBSERVATION_test = 500. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d_1(x,W):
    return tf.nn.conv2d(x, W, strides = [1,4,4,1], padding = 'SAME')

def conv2d_2(x,W):
    return tf.nn.conv2d(x, W, strides = [1,2,2,1], padding = 'SAME')

def conv2d_3(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool_2x2(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


def buildmodel():
    first_conv_net = tf.Graph()

    with first_conv_net.as_default():
        image = tf.placeholder(tf.float32,shape = (None, 80*80*4))
        x_image = tf.reshape(image, [-1,80,80,4])
        label = tf.placeholder(tf.float32, shape = (None,2))#not yet known

        #5x5 convolution layer, pool 2x2, depth 1 --> depth 32
        W_conv1 = weight_variable([8,8,4,32])
        b_conv1 = bias_variable([20,20,32])
        h_conv1 = tf.nn.relu(conv2d_1(x_image,W_conv1) + b_conv1)
        #h_pool1 = max_pool_2x2(h_conv1)

        #5x5 convolution layer, pool 2x2, depth 32 --> depth 64
        W_conv2 = weight_variable([4,4,32,64])
        b_conv2 = bias_variable([10,10,64])
        h_conv2 = tf.nn.relu(conv2d_2(h_conv1, W_conv2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        #5x5 convolution layer, pool 2x2, depth 32 --> depth 64
        W_conv3 = weight_variable([3,3,64,64])
        b_conv3 = bias_variable([10,10,64])
        h_conv3 = tf.nn.relu(conv2d_3(h_conv2, W_conv3) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv2)

        input_dim = 10*10*64
        #Flatten the filtered images in a vector
        h_conv3_flat = tf.reshape(h_conv3, [-1, input_dim])

        keep_prob = tf.placeholder(tf.float32)
        #Fully connected layer of 1024 neurons (activation function: ReLU) with dropout(prob = keep_prob)
        W_fc1 = weight_variable([input_dim,512])
        b_fc1 = bias_variable([512])
        h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #Fully connected layer into 2 labels (activation function: Linear)
        W_fc2 = weight_variable([512, 2])
        b_fc2 = bias_variable([2])
        scores = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #real value
        
        #Loss function: cross-entropy with softmax loss; numerically stable way
        loss = tf.losses.mean_squared_error(label,scores)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = scores))

        #Optimization algorithm: ADAM, see https://arxiv.org/abs/1412.6980
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        #Compute performance
        #correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(label,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    return first_conv_net

###CLASS LOOP##
## Size of the mini-batch
#mini_batch_size = 100
## Number of SGD steps
#n_steps = 1000

#training_data = {image: mnist.train.images, label: mnist.train.labels}
#validation_data = {image: mnist.validation.images, label: mnist.validation.labels}
#test_data = {image: mnist.test.images, label: mnist.test.labels}

#with tf.Session(graph=first_conv_net) as sess:
#    tf.global_variables_initializer().run()
#    # Training loop
#    for step in range(n_steps + 1):
#        # Get next mini-batch
#        batch_images, batch_labels = mnist.train.next_batch(mini_batch_size)
#        feed = {image: batch_images, label: batch_labels}
#        _, current_loss = sess.run([train_step, loss], feed_dict=feed)        
#        if step % 200 == 0:
#            print('Step %d' % step)
#            print('....Loss:     %f' % current_loss)
#            print('....Accuracy on train: %f' % sess.run(accuracy, feed_dict=training_data))
#            print('....Accuracy on validation: %f' % sess.run(accuracy, feed_dict=validation_data))
#    print('Accuracy on test: %f' % sess.run(accuracy, feed_dict=test_data))
#    # Save the weights into a numpy ndarray for plotting
#    weights = sess.run(W)
#
####QLEARN LOOP####

def trainNetwork(args):
    first_conv_net = tf.Graph()

    with first_conv_net.as_default():
        image = tf.placeholder(tf.float32,shape = (None, 80,80,4))
        x_image = image
        #x_image = tf.reshape(image, [-1,80,80,4])
        label = tf.placeholder(tf.float32, shape = (None,2))#not yet known

        #5x5 convolution layer, pool 2x2, depth 1 --> depth 32
        W_conv1 = weight_variable([8,8,4,32])
        b_conv1 = bias_variable([20,20,32])
        h_conv1 = tf.nn.relu(conv2d_1(x_image,W_conv1) + b_conv1)
        #h_pool1 = max_pool_2x2(h_conv1)

        #5x5 convolution layer, pool 2x2, depth 32 --> depth 64
        W_conv2 = weight_variable([4,4,32,64])
        b_conv2 = bias_variable([10,10,64])
        h_conv2 = tf.nn.relu(conv2d_2(h_conv1, W_conv2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        #5x5 convolution layer, pool 2x2, depth 32 --> depth 64
        W_conv3 = weight_variable([3,3,64,64])
        b_conv3 = bias_variable([10,10,64])
        h_conv3 = tf.nn.relu(conv2d_3(h_conv2, W_conv3) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv2)

        input_dim = 10*10*64
        #Flatten the filtered images in a vector
        h_conv3_flat = tf.reshape(h_conv3, [-1, input_dim])

        keep_prob = tf.placeholder(tf.float32)
        #Fully connected layer of 1024 neurons (activation function: ReLU) with dropout(prob = keep_prob)
        W_fc1 = weight_variable([input_dim,512])
        b_fc1 = bias_variable([512])
        h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #Fully connected layer into 2 labels (activation function: Linear)
        W_fc2 = weight_variable([512, 2])
        b_fc2 = bias_variable([2])
        scores = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #real value
        
        #Loss function: cross-entropy with softmax loss; numerically stable way
        #loss = tf.losses.mean_squared_error(label,scores)
        loss_value = tf.reduce_mean(tf.squared_difference(scores, label))
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = scores))

        #Optimization algorithm: ADAM, see https://arxiv.org/abs/1412.6980
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_value)


    with tf.Session(graph = first_conv_net) as sess:
        tf.global_variables_initializer().run()
        # open up a game state to communicate with emulator
        game_state = game.GameState()

        # store the previous observations in replay memory
        D = deque()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = game_state.frame_step(do_nothing)

        x_t = skimage.color.rgb2gray(x_t)
        x_t = skimage.transform.resize(x_t,(80,80))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        #print (s_t.shape)

        #In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

        if args['mode'] == 'Run':
            OBSERVE = 999999999    #We keep observe, never train
            epsilon = FINAL_EPSILON
            ##feed = {image: batch_images, label: batch_labels}
            ##_, current_loss = sess.run([train_step, loss], feed_dict=feed)        
            print ("Now we load weight")
            #model.load_weights("model.h5")
            #adam = Adam(lr=LEARNING_RATE)
            #model.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")    
        else:                       #We go to training mode
            OBSERVE = OBSERVATION_test
            epsilon = INITIAL_EPSILON

        t = 0
        while (True):
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            a_t = np.zeros([ACTIONS])
            #choose an action epsilon greedy
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t[action_index] = 1
                else:
                    feed = {image: s_t, keep_prob: 0.5}
                    q = sess.run(scores, feed_dict = feed) # TF_version
                    #print(sess.run(scores, feed_dict=feed)) 
                    #q = model.predict(s_t) #input a stack of 4 images, get the prediction_KERAS VERSION
                    max_Q = np.argmax(q)
                    action_index = max_Q
                    a_t[max_Q] = 1

            #We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            #run the selected action and observed next state and reward
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

            x_t1 = skimage.color.rgb2gray(x_t1_colored)
            x_t1 = skimage.transform.resize(x_t1,(80,80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            # store the transition in D
            D.append((s_t, action_index, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            #only train if done observing
            if t > OBSERVE:
                #sample a minibatch to train on
                minibatch = random.sample(D, BATCH)



                inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
                print (inputs.shape)
                targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

                #Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    # if terminated, only equals reward

                    inputs[i:i + 1] = state_t    #I saved down s_t

                    #feed = {image: batch_images, label: batch_labels}
                    #_, current_loss = sess.run([train_step, loss], feed_dict=feed)
                    #targets[i] = model.predict(state_t)  # Hitting each buttom probability - KERAS
                    #Q_sa = model.predict(state_t1) #KERAS
                    feed_state_t = {image:state_t, keep_prob: 0.5}
                    feed_state_t1 = {image:state_t1, keep_prob: 0.5}
                    targets[i] = sess.run(scores,feed_dict = feed_state_t) 
                    Q_sa = sess.run(scores,feed_dict = feed_state_t1)
                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

                #targets2 = normalize(targets)
                #loss += model.train_on_batch(inputs, targets)
                feed_train = {image: inputs, label: targets, keep_prob: 0.5}
                _,current_loss = sess.run([train_step,loss_value], feed_dict=feed_train)
                loss += current_loss
                    

            s_t = s_t1
            t = t + 1

            # save progress every 10000 iterations
            #if t % 1000 == 0:
             #   print("Now we save model")
              #  model.save_weights("model.h5", overwrite=True)
               # with open("model.json", "w") as outfile:
                #    json.dump(model.to_json(), outfile)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

        print("Episode finished!")
        print("************************")


def playGame(args):
    trainNetwork(args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    main()