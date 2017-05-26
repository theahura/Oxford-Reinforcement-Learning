"""
Author: Amol Kapoor
Description: Runner for slither.io AI.
"""

import logging
import sys

import numpy as np
import tensorflow as tf
from universe import spaces

import constants as c
import humantest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def convert_to_uni_action(left=False, right=False, space=False):
    """
    Goes from actions to universe actions.
    """
    return [spaces.KeyEvent.by_name('space', down=space),
            spaces.KeyEvent.by_name('left', down=left),
            spaces.KeyEvent.by_name('right', down=right)]

class PartialRollout(object):
    """
	a piece of a complete rollout.  We run our agent, and process its experience
	once it has processed enough steps. See starter agent.
	"""
    def __init__(self, worker_index):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.worker = worker_index
        self.terminal = False
        self.features = []
        self.total_reward = 0

    def add(self, state, action, reward, value, terminal, features):
        """
        Adds a new event to the rollout.
        """
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

def run_env(env, policy, worker_index, humantrain=False):
    """
    Runs the universe environment
    """

    sess = tf.get_default_session()
    last_state = env.reset()
    last_c_in, last_h_in = policy.get_initial_features()
    rewards = 0
    steps = 0

    while True:
        rollout = PartialRollout(worker_index)

        # Stop adding failed lives to the rollout
        action = np.array([1, 0, 0, 0, 0, 0])
        firstaction = action

        while True:
            logger.info("IN GAME")
            steps += 1

            # Get the action
            if not humantrain:
                output = policy.get_action(last_state, last_c_in, last_h_in)
                action, value, features = output[0], output[1], output[2:]
            else:
                output = policy.get_action(last_state, last_c_in, last_h_in)
                value, features = output[1], output[2:]
                if humantest.isData():
                    ch = sys.stdin.read(1)
                    action = np.array(humantest.convert_to_action(ch))
                else:
                    action = np.array(humantest.convert_to_action(None))

            final_action = action.argmax()

            if c.WORKER_DEBUG:
                logger.info("WORKER %d TRANSLATED ACTION: %s", worker_index,
                            str(c.ACTIONS[final_action]))

            left, right, up = c.ACTIONS[final_action]

            final_action = convert_to_uni_action(left, right, up)
            # Run the action
            state, reward, terminal, info = env.step(final_action)

            # Scale the reward but save the actual value
            rewards += reward

            if terminal:
                reward = c.END_GAME_REW
            elif reward == 0:
                reward = c.ZERO_REW_VAL
            else:
                reward = reward*c.REW_SCALE

            if c.WORKER_DEBUG:
                logger.info("WORKER %d REW: %s", worker_index, str(reward))
                logger.info("WORKER %d TERM: %s", worker_index, str(terminal))
                logger.info("WORKER %d INFO: %s", worker_index, str(info))

            # Process the action results
            rollout.add(last_state, action, reward, value, terminal,
                        (last_c_in, last_h_in))

            if c.WORKER_DEBUG:
                logger.info("WORKER %d TOTAL REW: %s", worker_index,
                            str(rewards))

            # Reset for next loop
            last_state = state
            last_c_in = features[0]
            last_h_in = features[1]

            # Check if we hit game over
            if terminal:
                rollout.total_reward = rewards
                rewards = 0
                last_c_in, last_h_in = policy.get_initial_features()
                break
            elif steps % c.ENV_STEPS == 0 and not c.HUMAN_TRAIN:
                rollout.total_reward = rewards
                break
            else:
                rollout.r = policy.value(last_state, last_c_in, last_h_in)

        yield rollout
