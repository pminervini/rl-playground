#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import gym
import numpy as np

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
logger.setLevel(logging.INFO)


def main(argv):
    # env = gym.make('FrozenLake-v0')
    env = gym.make('FrozenLake8x8-v0')

    num_episodes = 2000

    learning_rate = .8
    γ = 0.95

    logger.info('Observation space size: {}'.format(env.observation_space.n))
    logger.info('Action space size: {}'.format(env.action_space.n))

    # Initialize Q table with all zeros
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    logger.info('Q table shape: {}'.format(q_table.shape))

    reward_lst = []
    for i in range(num_episodes):

        s = env.reset()
        reward = 0

        for j in range(100):
            # Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(q_table[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

            # Get new state and reward from environment
            s1, r, d, _ = env.step(a)

            # Update Q-Table with new knowledge
            q_table[s, a] = q_table[s, a] + learning_rate * (r + γ * np.max(q_table[s1, :]) - q_table[s, a])

            s = s1
            reward += r

        logger.debug('Episode: {}\tReward: {:.4f}'.format(i, reward))
        reward_lst += [reward]

    logger.info('Reward over time: {:.4f}'.format(sum(reward_lst)/num_episodes))

    print(q_table)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

