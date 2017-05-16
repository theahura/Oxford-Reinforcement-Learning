"""
Author: Amol Kapoor
Description: Runner for slither.io AI.
"""

import time
import logging

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
    logger.info("Obs space: %s", str(env.observation_space.shape))
    policy = model.Policy(env.observation_space.shape, c.NUM_ACTIONS)

    x = 0
    y = 0
    click = 0
    start = time.time()

    init = tf.global_variables_initializer()

    with tf.Session() as sess, sess.as_default():
        sess.run(init)

        while True:
            if time.time() - start >= 3 and observation_n:
                start = time.time()

                c_in, h_in = policy.get_initial_features()

                output = policy.get_action(observation_n, c_in, h_in)
                actions = output[0]
                x = int(x*c.WINDOW_WIDTH + c.WINDOW_START[0])
                y = int(actions[1]*c.WINDOW_HEIGHT + c.WINDOW_START[1])
                click = 1 if actions[2] > 0.5 else 0
                logger.info("X Y CLICK: %s", str([x, y, click]))

            if c.DEBUG:
                logger.info("Observation before shape: %s ",
                            str(np.array(observation_n).shape))
                observation_n = np.reshape(observation_n, (128, 200))
                plt.imshow(observation_n)
                plt.savefig('test.png')

            action_n = [[universe.spaces.PointerEvent(x, y, click)]]

            observation_n, _, _, _ = env.step(action_n)


        env.render()

if __name__ == '__main__':
    run()
