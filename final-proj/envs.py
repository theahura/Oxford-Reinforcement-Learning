"""
Author: Amol Kapoor
Description: Sets up the environment
"""

import cv2
from gym.spaces.box import Box
import gym
from gym import spaces
import numpy as np
import logging
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, Logger, Vision, Unvectorize
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode

import constants as c

universe.configure_logging()

def create_env():
    """
    Creates the environment for slitherio.
    """
    env = gym.make(c.GAME_ID)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = CropScreen(env, c.WINDOW_HEIGHT, c.WINDOW_WIDTH, c.WINDOW_START[1],
                     c.WINDOW_START[0])
    env = Rescale(env)
    keys = c.ACTIONS
    env = DiscreteToFixedKeysVNCActions(env, keys)
    env = Unvectorize(env)
    env.configure(fps=c.FPS, remotes=1)
    return env

class CropScreen(vectorized.ObservationWrapper):
    """
	Crops out a [height]x[width] area starting from (top,left).
	Copied from starter agent.
	"""
    def __init__(self, env, height, width, top=0, left=0):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        return [ob[self.top:self.top+self.height,
                   self.left:self.left+self.width, :] if ob is not None
                else None for ob in observation_n]

def _process_frame(frame):
    """
    Processes a frame for slither. Copied from starter agent.
    """
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [c.RESIZE_X, c.RESIZE_Y, 1])
    return frame

class Rescale(vectorized.ObservationWrapper):
    """
    Rescales the environment observations. Copied from starter agent.
    """
    def __init__(self, env=None):
        super(Rescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [128, 200, 1])

    def _observation(self, observation_n):
        return [_process_frame(observation) for observation in
                observation_n]


class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n


class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-separated list of keys
    For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right'])
    will have 3 actions: [none, left, right]
    You can define a state with more than one key down by separating with spaces. For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right', 'space', 'left space', 'right space'])
    will have 6 actions: [none, left, right, space, left space, right space]
    """
    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self):
        self._actions = []
        uniq_keys = set()
        for key in self._keys:
            for cur_key in key.split(' '):
                uniq_keys.add(cur_key)

        for key in [''] + self._keys:
            split_keys = key.split(' ')
            cur_action = []
            for cur_key in uniq_keys:
                cur_action.append(
                    vnc_spaces.KeyEvent.by_name(cur_key,
                                                down=(cur_key in split_keys)))
            self._actions.append(cur_action)
        self.key_state = FixedKeyState(uniq_keys)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]
