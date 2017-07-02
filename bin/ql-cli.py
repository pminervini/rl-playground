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

    nb_episodes = 20000
    learning_rate = 0.5
    γ = 0.95

    # Initialize Q (state x action) table with all zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    logger.info('Q table shape: {}'.format(Q.shape))

    reward_lst = []

    for episode_idx in range(nb_episodes):
        # Observe the current state
        state = env.reset()

        sum_rewards = 0

        # Annealing schedule for the qty. of noise used when (greedily) selecting the next action
        ε = 1. / (episode_idx + 1.)

        for transition_idx in range(100):
            # Select an action by greedily (with noise) picking from the Q table
            noise = np.random.randn(1, env.action_space.n)
            a = np.argmax(Q[state, :] + ε * noise)

            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(a)

            # Update Q-Table with new knowledge
            Q[state, a] = Q[state, a] + learning_rate * (reward + γ * np.max(Q[new_state, :]) - Q[state, a])

            state = new_state
            sum_rewards += reward

            if done:
                break

        logger.info('Episode {}\t Reward: {}'.format(episode_idx, sum_rewards))
        reward_lst += [sum_rewards]

    logger.info('Reward over time: {:.4f}'.format(sum(reward_lst)/nb_episodes))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

