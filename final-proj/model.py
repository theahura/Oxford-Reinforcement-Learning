"""
Author: Amol Kapoor
Description: Tensorflow NN model for actions and updates.
See starter agent - this is heavily influenced (and in some cases taken
directly) from the implementation there, though this makes sense as the
framework is more or less the same but the variables can change.
"""

import logging
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import constants as c

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def normalized_columns_initializer(std=1.0):
    """
    Normalized initializer. See startup agent.
    """
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    """
    Flattens input across batches.
    """
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=1):
    """
    Defines 2d convolution layer. Influenced by starter agent.
    """
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        w_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]),
                   num_filters]
        b_shape = [1, 1, 1, num_filters]

        # initialize weights with random weights, see ELU paper
        # which cites He initialization
        # filter w * h * channels in
        fan_in = filter_size[0] * filter_size[1] * int(x.get_shape()[3])
        # filter w * h * channels out
        fan_out = filter_size[0] * filter_size[1] * num_filters

        w_bound = np.sqrt(12. / (fan_in + fan_out))

        w = tf.get_variable("W", w_shape, tf.float32,
                            tf.random_uniform_initializer(-w_bound, w_bound))

        b = tf.get_variable("b", b_shape,
                            initializer=tf.constant_initializer(0.0))

        return tf.nn.conv2d(x, w, stride_shape, padding="SAME") + b

def linear(x, size, name):
    """
    Defines a linear layer in tf.
    """
    logger.info("shape: %s", str(x.get_shape()))
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size])
    b = tf.get_variable(name + "/b", [size],
                        initializer=tf.constant_initializer(0))
    return tf.matmul(x, w) + b

class Policy(object):
    """
    NN that describes the policy for the agent.
    """

    def __init__(self, input_shape, action_size):
        """
        Sets up the model.
            Input - one frame, shape (batch=1, height, width, channel=1)
            Convnet - 32 channel out
            Flatten - allows for lstm over time (see starter agent)
            LSTM - 256 unit LSTM cell
            Linear - convert LSTM output to actions
        """
        # Need to add an extra dim to account for batch later
        self.x = x = tf.placeholder(tf.float32, [None] + list(input_shape))

        for i in xrange(c.CONV_LAYERS):
            x = tf.nn.elu(conv2d(x, 32, 'l{}'.format(i + 1), c.FILTER_SHAPE,
                                 c.STRIDE))

        # Add a time dimension
        x = tf.expand_dims(flatten(x), [0])

        # Now the LSTM layer
        cell = tf.contrib.rnn.LSTMCell(c.LSTM_UNITS, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, c.INPUT_KEEP_PROB,
                                             c.OUTPUT_KEEP_PROB)

        # Set up the initial state for episode resets later on
        self.c_init = np.zeros((1, cell.state_size.c), np.float32)
        self.h_init = np.zeros((1, cell.state_size.h), np.float32)

        # Need to hold the state of the LSTM for inputs from the previous frame
        self.c_in = tf.placeholder(tf.float32, [1, cell.state_size.c])
        self.h_in = tf.placeholder(tf.float32, [1, cell.state_size.h])
        state_in = rnn.LSTMStateTuple(self.c_in, self.h_in)

        outputs, (state_c, state_h) = tf.nn.dynamic_rnn(
            cell, x, initial_state=state_in,
            sequence_length=[tf.shape(self.x)[0]], time_major=False)

        # Make this a column vector to make the linear math easier
        x = tf.reshape(outputs, [-1, c.LSTM_UNITS])

        # As per A3C, logits and value function both are represented by linear
        # layers on top of the rest of the network
        self.logits = tf.squeeze(linear(x, action_size, "action"))

        # Also define the value function, for a single output
        self.vf = tf.reshape(linear(x, 1, "value"), [-1])

        # Other stuff
        self.state_out = [state_c[:1, :], state_h[:1, :]]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)

    def get_initial_features(self):
        """
        Gets the initial feature state for the lstm for episode reset.
        """
        return self.c_init, self.h_init

    def get_action(self, ob, c_in, h_in):
        """
        Returns the actions for a given state. Specifically, returns the x,y
        position for the mouse and whether or not lmb is clicked.
        """
        x = 0
        y = 0
        click = 0

        if c.DEBUG or c.RANDOM_POLICY:
            # Move randomly
            x = random.randint(c.WINDOW_START[0],
                               c.WINDOW_END[0])
            y = random.randint(c.WINDOW_START[1],
                               c.WINDOW_END[1])
            click = random.randint(0, 1)

            return [[x, y, click], random.randint(0, 100), [None]]

        sess = tf.get_default_session()
        return sess.run([self.logits, self.vf] + self.state_out,
                        {self.x: ob, self.c_in: c_in, self.h_in: h_in})

    def value(self, ob, c_in, h_in):
        """
        Runs the value function.
        """
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.c_in: c_in,
                                  self.h_in: h_in})[0]

    def update_model(self, reward):
        """
        Calculates gradients from the reward and updates the tf model
        accordingly.
        """
        pass
