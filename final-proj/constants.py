"""
Author: Amol Kapoor
Description: Constants for network and other parameters.
"""

# Slither.io params
#       Set in stone
GAME_ID = 'internet.SlitherIO-v0'
WINDOW_START = (30, 100)
WINDOW_END = (500, 370)
WINDOW_HEIGHT = WINDOW_END[1] - WINDOW_START[1]
WINDOW_WIDTH = WINDOW_END[0] - WINDOW_START[0]
#       Actions
ACTIONS = ['left', 'right', 'up', 'left up', 'right up']
NUM_ACTIONS = len(ACTIONS) # x coord, y coord, lmb click or not
#       Game score
ZERO_REW_VAL = 0 # Punish not getting pellets
REW_SCALE = 1
END_GAME_REW = -100 # Punish game overs
#       Optimizations
FPS = 5.0
RESIZE_X = 128
RESIZE_Y = 200
OBSERVATION_SPACE = (RESIZE_X, RESIZE_Y, 1)

# Network Params
#       Network Constructions
CONV_LAYERS = 4
FILTER_SHAPE = [3, 3]
STRIDE = 2
OUTPUT_CHANNELS = 32
LSTM_UNITS = 256
#       A3C Params
VF_LOSS_CONST = 0.5
ENT_CONST = 0.01
LEARNING_RATE = 0.1
LEARNING_RATE_STEP = 25
LEARNING_RATE_SCALE = 0.99
GAMMA = 0.99
LAMBDA = 1.0
#       Optimizations and Convergence Tricks
INPUT_KEEP_PROB = .5
OUTPUT_KEEP_PROB = .5
MAX_GRAD_NORM = 80.0

# Misc
STEPS_TO_SAVE = 5 # saves every n lives or updates (i.e. env_steps*n steps)
ENV_STEPS = 100 # Number of steps in runner before updating global
NUM_WORKERS = 2 # Number of threads to use, number of workers is actually n - 1
SLEEP_TIME = 300 # Wait 5 minutes before restarts

# Debugging
DEBUG = True
GLOBAL_DEBUG = True
MODEL_DEBUG = True
WORKER_DEBUG = True
RANDOM_POLICY = False
CKPT_PATH = 'data/ckpt/'
LOGDIR = 'data/logs/'
SUM_STEPS = 5
