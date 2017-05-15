"""
Author: Amol Kapoor
Description: Runner for slither.io AI.
"""

import time
import logging

import gym
import numpy as np
import matplotlib.pyplot as plt
import universe  # register the universe environments

import constants as c
import envs
import model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run():
    """ Runs the slitherio AI. """
    env = envs.create_env()
    observation_n = env.reset()
    policy = model.Policy(env.observation_space.shape, c.NUM_ACTIONS)

    x = 0
    y = 0
    click = 0
    start = time.time()

    while True:
        if time.time() - start >= 3 and observation_n:
            start = time.time()
            x, y, click = policy.get_action(observation_n)

            observation_n = np.reshape(observation_n, (128, 200))
            logger.info("Observation shape: %s ", str(observation_n.shape))
            plt.imshow(observation_n)
            plt.savefig('test.png')

        logger.info("Observation: %s", str(observation_n))
        logger.info("Observation shape: %s ", str(np.array(observation_n).shape))

        action_n = [[universe.spaces.PointerEvent(x, y, click)]]

        observation_n, reward_n, done_n, info = env.step(action_n)


        env.render()

if __name__ == '__main__':
    run()
