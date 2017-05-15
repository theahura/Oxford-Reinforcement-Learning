"""
Author: Amol Kapoor
Description: Tensorflow NN model for actions and updates.
See starter agent - this is heavily influenced (and in some cases taken
directly) from the implementation there, though this makes sense as the
framework is more or less the same but the variables can change.
"""

import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import constants as c

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
    Flattens input.
    """
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME",
           dtype=tf.float32, collections=None):
    """
    Defines 2d convolution layer. Taken from starter agent.
    """

    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], 1,
                        num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters

        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype,
                            tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)

        b = tf.get_variable("b", [1, 1, 1, num_filters],
                            initializer=tf.constant_initializer(0.0),
                            collections=collections)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    """
    Defines a linear layer in tf.
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size],
                        initializer=initializer)
    b = tf.get_variable(name + "/b", [size],
                        initializer=tf.constant_initializer(bias_init))
    return tf.matmul(w, b)

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
        self.x = x = tf.placeholder(tf.float32, [None] + list(input_shape))

        for i in xrange(c.CONV_LAYERS):
            x = tf.nn.elu(conv2d(x, 32, 'l{}'.format(i + 1), c.FILTER_SHAPE,
                                 c.STRIDE_SHAPE))

        x = tf.expand_dims(flatten(x), [0])

        lstm = tf.contrib.rnn.LSTMCell(c.LSTM_UNITS, state_is_tuple=True)
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, c.INPUT_KEEP_PROB,
                                             c.OUTPUT_KEEP_PROB)

        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, c.LSTM_UNITS])
        self.logits = linear(x, action_size, "action",
                             normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value",
                                    normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, action_size)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)

    def get_initial_features(self):
        """
        Gets the initial feature state for the lstm.
        """
        return self.state_init

    def get_action(self, ob, c_in, h):
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

            return x, y, click

        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c_in,
                         self.state_in[1]: h})

    def value(self, ob, c_in, h):
        """
        Runs the value function.
        """
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c_in,
                                  self.state_in[1]: h})[0]

    def update_model(self, reward):
        """
        Calculates gradients from the reward and updates the tf model
        accordingly.
        """
        pass
