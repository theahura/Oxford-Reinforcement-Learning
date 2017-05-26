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
ACTIONS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1)]
NUM_ACTIONS = len(ACTIONS)
#       Reward function
ZERO_REW_VAL = 0 # Punish not growing
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
ADAM = True # Whether to use adamopt or grad descent
#       A3C Params
VF_LOSS_CONST = 0.5
ENT_CONST = 0.01 # Encourage exploration with more entropy
LEARNING_RATE = 0.1
LEARNING_RATE_STEP = 25
LEARNING_RATE_SCALE = 0.99
GAMMA = 0.99 # Discount past rewards by this much
LAMBDA = 1.0
#       Optimizations and Convergence Tricks
INPUT_KEEP_PROB = 0.5 # Dropout for lstm
OUTPUT_KEEP_PROB = 0.5
CONV_KEEP_PROB = 0.5 # Dropout for convnet
MAX_GRAD_NORM = 60.0
REG_CONST = 0.0001 # L2 norm regularization

# Debugging
DEBUG = True
GLOBAL_DEBUG = False
MODEL_DEBUG = False
WORKER_DEBUG = False
RANDOM_POLICY = False
CKPT_PATH = 'data/ckpt/'
LOGDIR = 'data/logs/'
SUM_STEPS = 1

# Misc
STEPS_TO_SAVE = 5 # saves every n lives or updates (i.e. env_steps*n steps)
ENV_STEPS = 100.0*FPS # Number of steps in runner before updating global
SLEEP_TIME = 60 # Wait 5 minutes before restarts

# Args
HUMAN_TRAIN = True # Learn over the shoulder
ASYNC_HUMAN_TRAIN = False
PLAY = False # Play only global
NUM_WORKERS = 1 if PLAY or HUMAN_TRAIN else 2 if ASYNC_HUMAN_TRAIN else 6
