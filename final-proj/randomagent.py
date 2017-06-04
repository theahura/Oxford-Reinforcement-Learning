import random

import gym
import universe  # register the universe environments
from universe import spaces
from universe.wrappers import BlockingReset, Logger, Vision, Unvectorize

import constants as c
import runner

env = gym.make('internet.SlitherIO-v0')
env = BlockingReset(env)
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

total_reward = 0
while True:
    action = random.choice(c.ACTIONS)
    action_n = runner.convert_to_uni_action(action[0], action[1], action[2])
    observation_n, reward_n, done_n, info = env.step([action_n])
    total_reward += reward_n[0]
    if done_n[0]:
        print 'Total reward %d' % total_reward
        total_reward = 0
