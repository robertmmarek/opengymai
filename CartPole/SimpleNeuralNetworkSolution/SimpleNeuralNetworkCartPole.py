import gym
import tensorflow as tf
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1
learning_rate = 0.001

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer_1 = tf.layers.dense(X,
                                 num_hidden,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer)

hidden_layer_2 = tf.layers.dense(hidden_layer_1,
                                 num_hidden,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer)

output_layer = tf.layers.dense(hidden_layer_2,
                               num_outputs,
                               activation=tf.nn.sigmoid,
                               kernel_initializer=initializer)

probabilities = tf.concat(axis=1, values=[output_layer, 1-output_layer])
action = tf.multinomial(probabilities, num_samples=1)

reward_ph = tf.placeholder(tf.float32, shape=[None, 2])
loss = tf.reduce_mean(tf.square(reward_ph-output_layer))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epi = 250
step_limit = 500
env = gym.make('CartPole-v0')
avg_steps = []

with tf.Session() as sess:
    init.run()

    for i_episode in range(epi):
        obs = env.reset()

        for step in range(step_limit):
            action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
            train_obs = obs

            obs, reward, done, info = env.step(action_val[0][0])

            train_reward = [0., 0.]
            train_reward[int(action_val)] = reward
            train_reward[1-int(action_val)] = 1.-reward
            sess.run(train, feed_dict={X: train_obs.reshape(1, num_inputs),
                                       reward_ph: np.array([train_reward])})

            if done:
                avg_steps.append(step)
                print("Done after {} steps".format(step))
                break

    print("After {} episodes steps per game was {}".format(epi,
                                                           np.mean(avg_steps)))
    env.close()

