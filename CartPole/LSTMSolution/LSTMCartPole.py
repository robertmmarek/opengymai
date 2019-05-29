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
lstm_rnn = tf.keras.layers.RNN(lstm_cell, return_sequences=False, return_state=True)
lstm_state_ph = [tf.placeholder(tf.float32, [None, lstm_size]), tf.placeholder(tf.float32, [None, lstm_size])]
lstm_outputs = lstm_rnn(inputs, initial_state=lstm_state_ph)
lstm_states = lstm_outputs[1::]
lstm_outputs = lstm_outputs[0]

logits = tf.keras.layers.Dense(number_of_logits, activation='sigmoid')(lstm_outputs)
selection = tf.cast(tf.random.multinomial(logits, 1), tf.float32)

labels = tf.reshape(tf.stack([1.-selection, selection], axis=1), shape=[-1, 2])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradient_multiplier = tf.placeholder(tf.float32, shape=[1])
gradients = optimizer.compute_gradients(loss)
modified_gradients = [(tf.multiply(tup[0], gradient_multiplier), tup[1]) for tup in gradients]
train = optimizer.apply_gradients(modified_gradients)

zero_states = [[[0.]*lstm_size], [[0]*lstm_size]]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    states = zero_states
    out, states = sess.run([selection, lstm_states], feed_dict={inputs: [[[1]]], lstm_state_ph[0]: states[0], lstm_state_ph[1]: states[1]})
    print(states)
    out, states = sess.run([selection, lstm_states], feed_dict={inputs: [[[1]]], lstm_state_ph[0]: states[0], lstm_state_ph[1]: states[1]})
    print(states)
    