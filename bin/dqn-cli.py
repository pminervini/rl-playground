#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import gym

import numpy as np
import tensorflow as tf

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
logger.setLevel(logging.INFO)


def main(argv):
    env = gym.make('FrozenLake-v0')

    observation_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    logger.info('Observation space size: {}'.format(observation_space_size))
    logger.info('Action space size: {}'.format(action_space_size))

    # Feed-forward network used to choose actions
    inputs = tf.placeholder(shape=[1, observation_space_size], dtype=tf.float32, name='inputs')

    import inspect
    spec = inspect.getfullargspec(tf.contrib.layers.fully_connected)
    print(spec)

    Q_out = tf.contrib.layers.fully_connected(inputs=inputs,
                                              num_outputs=action_space_size,
                                              weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                              biases_initializer=tf.zeros_initializer(),
                                              activation_fn=None)
    predict = tf.argmax(Q_out, 1)

    # Loss: sum of squares difference between the target and prediction Q values
    next_Q = tf.placeholder(shape=[1, action_space_size], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_Q - Q_out))

    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model_op = trainer.minimize(loss)

    init_op = tf.global_variables_initializer()

    num_episodes = 2000

    γ = 0.99
    ε = 0.1

    def to_fd(s):
        return {inputs: np.identity(observation_space_size)[s:s + 1]}

    reward_lst = []

    with tf.Session() as session:
        session.run(init_op)

        for i in range(num_episodes):
            # Reset the environment, and get the first observation
            s = env.reset()
            sum_rewards = 0

            for j in range(100):
                # Choose an action from the Q-network
                a, all_Q = session.run([predict, Q_out], feed_dict=to_fd(s))

                # Chance of random action
                if np.random.rand(1) < ε:
                    a[0] = env.action_space.sample()

                # Execute the action, and get new state and reward from environment
                s1, r, d, _ = env.step(a[0])

                # Obtain the Q' values by feeding the new state through our network
                Q1 = session.run(Q_out, feed_dict=to_fd(s1))

                max_Q1 = np.max(Q1)
                target_Q = all_Q

                target_Q[0, a[0]] = r + γ * max_Q1

                # Train our network using target and predicted Q values
                fd = to_fd(s)
                fd.update({next_Q: target_Q})
                _ = session.run([update_model_op], feed_dict=fd)

                sum_rewards += r
                s = s1

            logger.info(sum_rewards)
            reward_lst += [sum_rewards]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

