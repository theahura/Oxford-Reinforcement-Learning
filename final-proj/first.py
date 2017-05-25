import gym
import universe  # register the universe environments
from universe import spaces

env = gym.make('internet.SlitherIO-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

while True:
    action_n = [[spaces.KeyEvent.by_name('left', down=False)] for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)
    print "THIS IS THE INFO"
    print "THIS IS THE INFO"
    print "THIS IS THE INFO"
    print "THIS IS THE INFO"
    print "THIS IS THE INFO"
    print "THIS IS THE INFO"
    print info
    env.render()
