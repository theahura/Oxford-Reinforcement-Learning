"""
Author: Amol Kapoor
Description: Tensorflow NN model for actions and updates.
"""

import numpy as np
import random
import tensorflow as tf

import constants as c


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME",
           dtype=tf.float32, collections=None):
    """
    Defines 2d convolution layer.
    """

    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]),
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

    def __init__(self, input_size, action_size, action_set):
        pass

    def get_action(self, ob):
        """
        Returns the actions for a given state. Specifically, returns the x,y
        position for the mouse and whether or not lmb is clicked.
        """
        x = 0
        y = 0
        click = 0

        if c.DEBUG:
            x = random.randint(24, 521)
            y = random.randint(85, 384)
            click = random.randint(0, 1)

        return x, y, click

    def update_model(self, reward):
        """
        Calculates gradients from the reward and updates the tf model
        accordingly.
        """
