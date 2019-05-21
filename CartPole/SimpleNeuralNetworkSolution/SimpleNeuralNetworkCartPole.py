import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as pp


# actions: 0 left, 1 right
# We will try to predict score of input, where input is both observation and action

# network structure
input_size = 5
hidden_size = 4
output_size = 1
learning_rate = 0.03

input = tf.placeholder(tf.float32, shape=[None, input_size])
hid_1 = tf.layers.dense(input, units=hidden_size, activation=tf.nn.relu)
output = tf.layers.dense(hid_1, units=output_size, activation=None)

y_reward = tf.placeholder(tf.float32, shape=[None, 1])

loss = tf.reduce_mean(tf.square(y_reward-output))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
initializer = tf.global_variables_initializer()


def decide_move(sess, obs):
    left_option = list(obs)+[0.]
    right_option = list(obs)+[1.]

    out = sess.run(output, feed_dict={input: [left_option, right_option]})
    return out, np.argmin(out.reshape(2))
    
# training params

max_moves_per_game = 50
training_steps = 500
games_to_play = 500
policy_gradient_steps = 1
policy_gradient_discount_rate = 0.3


def get_training_data(obs, moves, rewards, estimations):
    X = []
    Y_ = []
    Y = []
    for i in range(len(rewards)-policy_gradient_steps):
        sel_obs = obs[i]
        sel_moves = moves[i]
        x = list(sel_obs)+[float(sel_moves)]
        sel_rewards = rewards[i:i+5]
        y_ = [np.sum([val*np.power(policy_gradient_discount_rate, index) for index, val in enumerate(sel_rewards)])]
        y = [estimations[i]]

        X.append(x)
        Y_.append(y_)
        Y.append(y)

    return X, Y_, Y

with tf.Session() as sess:
    sess.run(initializer)
    env = gym.make('CartPole-v1')
    
    training_X = []
    training_Y = []
    for step in range(games_to_play):
        obs = env.reset()
        
        this_game_obs = []
        this_game_moves = []
        this_game_rewards = []
        this_game_estimations = []
        for _ in range(max_moves_per_game):
            this_game_obs.append(obs)
            out_est, move = decide_move(sess, obs)
            move = env.action_space.sample()
            this_game_estimations.append(out_est[move][0])
            this_game_moves.append(move)
            obs = env.step(move)
            reward = obs[0][2]#float(obs[1])
            is_end = obs[2]
            obs = obs[0]
            this_game_rewards.append(reward)

        X, Y_, Y = get_training_data(this_game_obs, this_game_moves, this_game_rewards, this_game_estimations)
        training_X += X
        training_Y += Y_

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(training_X, training_Y, test_size=0.2)

    for i in range(training_steps):
        sess.run(train, feed_dict={input: X_train, y_reward: y_train})
        res_train = sess.run(loss, feed_dict={input: X_train, y_reward: y_train})
        res_test = sess.run(loss, feed_dict={input: X_test, y_reward: y_test})
        print("train_loss: {} test_loss: {} %: {}".format(res_train, res_test, i/training_steps))
        
    number_of_show_plays = 10

    for play in range(number_of_show_plays):
        obs = env.reset()
        for _ in range(100):
            env.render()
            out_est, move = decide_move(sess, obs)
            obs = env.step(move)
            is_end = obs[2]

            if is_end:
                break
            
            obs = obs[0]

        print(_)




