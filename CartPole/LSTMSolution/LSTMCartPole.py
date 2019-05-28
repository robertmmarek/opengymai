import gym
import tensorflow as tf
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import gym

#network parameters
number_of_inputs = 1
lstm_size = 5
number_of_logits = 2
number_of_time_steps = 1
learning_rate = 0.001

#define network
inputs = tf.keras.layers.Input(shape=(None, number_of_inputs))
lstm_cell = tf.keras.layers.LSTMCell(units=lstm_size)
lstm_rnn = tf.keras.layers.RNN(lstm_cell, return_sequences=False)
lstm_outputs = lstm_rnn(inputs)
logits = tf.keras.layers.Dense(number_of_logits, activation='sigmoid')(lstm_outputs)
selection = tf.cast(tf.random.multinomial(logits, 1), tf.float32)

train_selection = tf.placeholder(dtype=tf.float32, shape=[None, 1])
labels = tf.reshape(tf.stack([1.-train_selection, train_selection], axis=1), shape=[-1, 2])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(loss)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    ret = sess.run(selection, feed_dict={inputs: [[[1], [2]]]})
    print(ret)
    