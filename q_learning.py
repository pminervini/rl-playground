#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import gym
import numpy as np

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    env = gym.make('FrozenLake-v0')

    num_episodes = 2000

    learning_rate = .8
    γ = 0.95

    # Initialize table with all zeros
    Q = np.zeros([env.observation_space.n,env.action_space.n])

    for i in range(num_episodes):
        s = env.reset()

        for j in range(100):
            # Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

            # Get new state and reward from environment
            s1, r, d, _ = env.step(a)

            # Update Q-Table with new knowledge
            Q[s, a] = Q[s, a] + learning_rate * (r + γ * np.max(Q[s1, :]) - Q[s, a])
            s = s1
    
    print(Q)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

