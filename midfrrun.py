import numpy as np
import tensorflow as tf
#import realenv as cbcenv
import realenvusb as cbcenv
env = cbcenv.ENV()

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

s = env.reset()



input_data_s = tf.keras.Input(shape=(s_dim,))
net = tf.keras.layers.Dense(100, activation=tf.nn.relu)(input_data_s)
net = tf.keras.layers.Dense(100, activation=tf.nn.relu)(net)
a = tf.keras.layers.Dense(a_dim, activation=tf.nn.tanh)(net)
output = tf.keras.layers.Lambda(lambda x: x*a_bound)(a)
model = tf.keras.Model(inputs=input_data_s, outputs=output)

model.load_weights("midfrusb-real.h5")

while True:
    a = model.predict(s.reshape(1,20))
    s, r, done, info = env.step(a)



