"""
Author: Amol Kapoor
Description: Sets up the environment
"""

import cv2
from gym.spaces.box import Box
import gym
import numpy as np
import logging
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, Logger, Vision

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
