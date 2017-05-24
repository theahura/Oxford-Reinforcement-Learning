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
ACTIONS = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1)]
NUM_ACTIONS = len(ACTIONS)
#       Reward function
ZERO_REW_VAL = -2 # Punish not growing
REW_SCALE = 1
END_GAME_REW = 0 # Punish game overs
#       Optimizations
FPS = 10.0
RESIZE_X = 128
RESIZE_Y = 200
OBSERVATION_SPACE = (RESIZE_X, RESIZE_Y, 1)

# Network Params
#       Network Constructions
CONV_LAYERS = 5
FILTER_SHAPE = [3, 3]
STRIDE = 2
OUTPUT_CHANNELS = 32
LSTM_UNITS = 512
#       A3C Params
VF_LOSS_CONST = 0.5
ENT_CONST = 0.01
LEARNING_RATE = 0.1
LEARNING_RATE_STEP = 25
LEARNING_RATE_SCALE = 0.99
GAMMA = 0.99
LAMBDA = 1.0
#       Optimizations and Convergence Tricks
INPUT_KEEP_PROB = 0.5
OUTPUT_KEEP_PROB = 0.5
MAX_GRAD_NORM = 60.0
REG_CONST = 0.000001

# Misc
STEPS_TO_SAVE = 5 # saves every n lives or updates (i.e. env_steps*n steps)
ENV_STEPS = 100.0*FPS # Number of steps in runner before updating global
NUM_WORKERS = 1 # Number of threads to use, number of workers is actually n - 1
SLEEP_TIME = 300 # Wait 5 minutes before restarts

# Debugging
DEBUG = True
GLOBAL_DEBUG = True
MODEL_DEBUG = False
WORKER_DEBUG = True
RANDOM_POLICY = False
CKPT_PATH = 'data/ckpt/'
LOGDIR = 'data/logs/'
SUM_STEPS = 5
