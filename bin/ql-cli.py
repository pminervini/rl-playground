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
    env = gym.make('FrozenLake-v0')

    observation_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    logger.info('Observation space size: {}'.format(observation_space_size))
    logger.info('Action space size: {}'.format(action_space_size))

    num_episodes = 20000
    learning_rate = 0.5
    γ = 0.95

    # Initialize Q table with all zeros
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    logger.info('Q table shape: {}'.format(q_table.shape))

    reward_lst = []
    for i in range(num_episodes):

        state = env.reset()
        sum_rewards = 0

        for j in range(100):
            # Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(q_table[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(a)

            # Update Q-Table with new knowledge
            q_table[state, a] = q_table[state, a] + learning_rate * (reward + γ * np.max(q_table[new_state, :]) - q_table[state, a])

            state = new_state
            sum_rewards += reward

            if done:
                break

        logger.info('Episode {}\t Reward: {}'.format(i, sum_rewards))
        reward_lst += [sum_rewards]

    logger.info('Reward over time: {:.4f}'.format(sum(reward_lst)/num_episodes))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

