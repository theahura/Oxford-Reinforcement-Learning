"""
Author: Amol Kapoor
Description: Runner for slither.io AI.
"""

import time
import logging
import os

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
logger.info("IN GAME")

def get_model(session, env):
    """
    Loads the tf model or inits a new one.
    """
    policy = model.Policy(env.observation_space.shape, c.NUM_ACTIONS)

    ckpt = tf.train.get_checkpoint_state(c.CKPT_PATH)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("LOADING OLD MODEL")
        policy.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("LOADING NEW MODEL")
        session.run(tf.global_variables_initializer())
    return policy

def run():
    """ Runs the slitherio AI. """
    logger.info("STARTING FROM THE TOP")
    env = envs.create_env()
    observation_n = env.reset()

    checkpoint_path = os.path.join(c.CKPT_PATH, 'slither.ckpt')

    start = time.time()

    last_ob = 0
    reward = [0]
    total_reward = 0
    last_value = 0
    last_state = (0, 0)
    done = False
    adv = 0.0

    x = 0
    y = 0
    click = 0

    logger.info("STARTING FROM THE TF")
    with tf.Session() as sess, sess.as_default():

        policy = get_model(sess, env)

        while True:

            logger.info("STILL IN THE WHILE LOOP")
            if time.time() - start >= 1 and observation_n:
                start = time.time()

                if last_ob:
                    adv = reward[0] + c.GAMMA * last_value
                    if done:
                        logger.info("FUCKED UP")
                        total_reward = -100000
                    policy.update_model(last_ob, last_state[0], last_state[1],
                                        adv, [total_reward])

                c_in, h_in = policy.get_initial_features()

                output = policy.get_action(observation_n, c_in, h_in)
                x, y, click = output[0]
                x = int(x*c.WINDOW_WIDTH + c.WINDOW_START[0])
                y = int(y*c.WINDOW_HEIGHT + c.WINDOW_START[1])
                click = 1 if click > 0.5 else 0

                last_ob = observation_n
                last_value = output[1]
                last_state = (c_in, h_in)
                logger.info("RESETTNIG TOTAL REWARD AND SAVING")
                total_reward = -10.0

                policy.saver.save(sess, checkpoint_path)

            if c.DEBUG:
                logger.info("Observation before shape: %s ",
                            str(np.array(observation_n).shape))
                observation_n = np.reshape(observation_n, (128, 200))
                plt.imshow(observation_n)
                plt.savefig('test.png')

            action_n = [[universe.spaces.PointerEvent(x, y, click)]]

            observation_n, reward, [done], _ = env.step(action_n)

            total_reward += (reward[0]*10)

            logger.info("DEBUGGING STUFF REW: %s", str(reward))
            logger.info("DEBUGGING STUFF TOT: %s", str(total_reward))
            logger.info("DEBUGGING STUFF DONE: %s", str(done))
            logger.info("DEBUGGING STUFF POS: %s", str((x, y, click)))
        env.render()

if __name__ == '__main__':
    run()
