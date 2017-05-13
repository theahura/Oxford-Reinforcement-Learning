"""
Author: Amol Kapoor
Description: Runner for slither.io AI.
"""

import gym
import numpy as np
import logging
import time
import universe  # register the universe environments

import constants as c
import model

env = gym.make('internet.SlitherIO-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

policy = model.Policy(0, 0, 0)

x = 0
y = 0
click = 0
start = time.time()

while True:
    if time.time() - start >= 3 and observation_n[0]:
        ob = observation_n[0]['vision']
        mask = np.all(np.equal(ob, 0), axis = 1)
        logger.info("Remove 0s: %s", str(ob[~mask].shape))
        start = time.time()
        x, y, click = policy.get_action(ob[~mask])

    action_n = [[universe.spaces.PointerEvent(x, y, click)] for ob in observation_n]

    observation_n, reward_n, done_n, info = env.step(action_n)

    env.render()

