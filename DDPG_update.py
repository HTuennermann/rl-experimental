"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time

#import cbcenv
import realenv as cbcenv

#####################  hyper parameters  ####################

MAX_EPISODES = 1000
MAX_EP_STEPS = 500
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 20000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        print(a_bound)
        #raise ValueError()
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a_feed = tf.placeholder(tf.float32, [None, a_dim], 'a_feed')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            print(self.a_.output[1])
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a.output[1], scope='eval', trainable=True)
            self.q_ = self._build_c(self.S_, self.a_.output[1], scope='target', trainable=False)

        # networks parameters
        self.ae_params = self.a.trainable_variables
        self.at_params = self.a_.trainable_variables
        
        self.ce_params = self.q.trainable_variables
        self.ct_params = self.q_.trainable_variables #tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * self.q_([self.S_, self.a_(self.S_)])
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        q_tensor = self.q([self.S, self.a(self.S)])
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q_tensor)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q_tensor)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a(self.S), {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)





        #self.q_.set_weights(self.q.get_weights())
        

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a(self.S): ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):

        input_data_s = tf.keras.Input(shape=s.shape)
        net = tf.keras.layers.Dense(30, activation=tf.nn.relu)(input_data_s)
        net = tf.keras.layers.Dense(30, activation=tf.nn.relu)(net)
        a = tf.keras.layers.Dense(1, activation=tf.nn.tanh)(net)
        output = tf.keras.layers.Lambda(lambda x: x*self.a_bound)(a)
        return tf.keras.Model(inputs=input_data_s, outputs=output)


    def _build_c(self, s, a, scope, trainable):

        n_l1 = 30
        #w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
        #w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
        #b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
        input_data_s = tf.keras.Input(shape=s.shape)
        input_data_a = tf.keras.Input(shape=a.shape)

        merged = tf.keras.layers.concatenate([input_data_s,input_data_a])
        net = tf.keras.layers.Dense(n_l1, activation=tf.nn.relu)(merged)
        net = tf.keras.layers.Dense(n_l1, activation=tf.nn.relu)(net)
        output = tf.keras.layers.Dense(1)(net)
            
        net = tf.layers.dense(tf.concat([s,a], axis=1), n_l1, tf.nn.relu)
            
        #net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
        return tf.keras.Model(inputs=[input_data_s,input_data_a], outputs=output)

###############################  training  ####################################

#env = gym.make(ENV_NAME)
#env = env.unwrapped
#env.seed(1)
env = cbcenv.ENV()

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 0.1  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -0.01, 0.01)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r , s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .99999    # decay the action randomness
            ddpg.learn()

        s = s_
        if j>100:
            ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %f' % (ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
ddpg.a.save_weights("ddpg2.h5")
print('Running time: ', time.time() - t1)
