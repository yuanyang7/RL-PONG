import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import cv2
import random
from collections import deque
import time

MAX_EPISODE = 1
GAMMA = 0.99 #discount factor for reward
DECAY_RATE = 0.99
ACTIONS = 3
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
def chooseAction(epsilon,observation_stack,sess,s,readout):            
    if np.random.rand() <= epsilon:
        action = np.random.choice([0,1,2])#?
    else:
        actions_value = readout.eval(feed_dict={s: [observation_stack]})[0]#todo
        action = np.argmax(actions_value) 
    return action
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def makeLayerVariables(self, shape, trainable, name_suffix):
        if self.normalizeWeights:
            # This is my best guess at what DeepMind does via torch's Linear.lua and SpatialConvolution.lua (see reset methods).
            # np.prod(shape[0:-1]) is attempting to get the total inputs to each node
            stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
            weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
            biases  = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
        else:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), trainable=trainable, name='W_' + name_suffix)
            biases  = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name='W_' + name_suffix)
        return weights, biases
def convertScreens(Screens):
    Screens = cv2.cvtColor(cv2.resize(Screens, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, Screens = cv2.threshold(Screens,1,255,cv2.THRESH_BINARY)    
    return Screens
def playGame(s, readout, h_fc1, sess):
    # define the cost function ?????????todo
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
 


    env = gym.make("Pong-v0")



    sess.run(tf.global_variables_initializer())
    memory = deque(maxlen = 2000)

    # printing
    a_file = open("logs_" + "pong" + "/readout.txt", 'w')
    h_file = open("logs_" + "pong" + "/hidden.txt", 'w')

    observation = env.reset() #get image
    time.sleep(0.1)
    observation = convertScreens(observation)
    observation_stack = np.asarray([observation,observation,observation,observation]) #80 80 4
    observation_stack = np.swapaxes(observation_stack,0,2)    
    
    #check point
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir = "./saved_networks", latest_filename = "pong-dqn-1380")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print ("Could not find old network weights")
    step = 0#todo
    epsilon = INITIAL_EPSILON
    while "RL" != "LR":

        while True:     
            env.render()
            #choose action
            action = chooseAction(epsilon,observation_stack,sess,s,readout)
                #decay epsilon
            if epsilon > FINAL_EPSILON:
                epsilon *= DECAY_RATE
            #return reward, next s
            next_state, reward, done, _ = env.step(action + 1)
            next_state = convertScreens(next_state) #80,80
            next_state = np.reshape(next_state, (1,80,80))
            next_state = np.swapaxes(next_state,0,2)
            observation_stack_ = np.append(observation_stack[:,:,1:4], next_state, axis = 2)
            #store
            action_store = np.zeros(ACTIONS)
            action_store[action] = 1
            memory.append((observation_stack, action_store, reward, observation_stack_, done))
            #train
            if (step > 200) and (step % 5 == 0):
                        # sample a minibatch to train on
                minibatch = random.sample(memory, BATCH_SIZE)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                for i in range(0, len(minibatch)):
                    # if terminal only equals reward
                    if minibatch[i][4]:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i])) #todo: ? not reward

                # perform gradient step\
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch})

            observation_stack = observation_stack_
            step  = step + 1
            print ("TIMESTEP", step,  "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", reward)
            if step % 10000 == 0:
                saver.save(sess, 'saved_networks/' + 'pong' + '-dqn' + str(step), global_step = step)

            #if teminate?
            if done:
                observation = env.reset()
                continue




if __name__ == "__main__":
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    playGame(s, readout, h_fc1, sess)

    playPong()