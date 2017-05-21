"""
Author: Amol Kapoor
Description: Runner for slither.io AI.
"""

import logging

import tensorflow as tf
import universe  # register the universe environments

import constants as c

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def run_env(env, policy, worker_index):
    sess = tf.get_default_session()
    last_state = env.reset()
    last_c_in, last_h_in = policy.get_initial_features()
    rewards = 0
    steps = 0

    while True:
        rollout = PartialRollout(worker_index)

        while True:

            steps += 1

            # Get the action
            output = policy.get_action(last_state, last_c_in, last_h_in)
            action, value, features = output[0], output[1], output[2:]
            translated_action = c.ACTIONS[action.argmax()]
            x_y = translated_action[0]
            x, y = x_y
            click = translated_action[1]
            translated_action = [universe.spaces.PointerEvent(x, y, click)]

            if c.WORKER_DEBUG:
                logger.info("WORKER %d TRANSLATED ACTION: %s", worker_index,
                            str(translated_action))
            # Run the action
            state, reward, terminal, info = env.step(translated_action)

            if c.WORKER_DEBUG:
                logger.info("WORKER %d REW: %s", worker_index, str(reward))
                logger.info("WORKER %d TERM: %s", worker_index, str(terminal))

            # Process the action results
            rollout.add(last_state, action, reward, value, terminal,
                        (last_c_in, last_h_in))
            rewards += reward

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
            elif steps % c.ENV_STEPS == 0:
                rollout.total_reward = rewards
                break
            else:
                rollout.r = policy.value(last_state, last_c_in, last_h_in)

        yield rollout
