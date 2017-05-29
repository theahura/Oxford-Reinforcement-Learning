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

def get_model(session, scope, index=None):
    """
    Loads the tf model or inits a new one.
    """
    policy = Policy(c.OBSERVATION_SPACE, c.NUM_ACTIONS, scope, index)

    ckpt = tf.train.get_checkpoint_state(c.CKPT_PATH)

    if scope == 'global':
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logger.info("GOT OLD MODEL FOR SCOPE %s", scope)
            policy.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            logger.info("STARTING NEW MODEL FOR SCOPE %s", scope)
            session.run(tf.global_variables_initializer())
    else:
        logger.info("STARTING NEW MODEL FOR SCOPE %s", scope)
    return policy

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

def linear(x, size, name, initializer):
    """
    Defines a linear layer in tf.
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size],
                        initializer=initializer)
    b = tf.get_variable(name + "/b", [size],
                        initializer=tf.constant_initializer(0))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    """
    Gets the actual output id using a multinomial sample based on the
    likelihoods of each action.
    """
    value = tf.squeeze(tf.multinomial(
        logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

class Policy(object):
    """
    NN that describes the policy for the agent.
    """

    def __init__(self, input_shape, action_size, scope, worker_index):
        """
        Sets up the model.
            Input - one frame, shape (batch=1, height, width, channel=1)
            Convnet - 32 channel out
            Flatten - allows for lstm over time (see starter agent)
            LSTM - 256 unit LSTM cell
            Linear - convert LSTM output to actions
        """
        # Need to add an extra dim to account for batch later
        self.x = x = tf.placeholder(tf.float32, [None] + list(input_shape),
                                    name="x")

        for i in xrange(c.CONV_LAYERS):
            x = tf.nn.elu(conv2d(x, c.OUTPUT_CHANNELS, 'l{}'.format(i + 1),
                                 c.FILTER_SHAPE, c.STRIDE))
            x = tf.nn.dropout(x, c.CONV_KEEP_PROB)

        # Add a time dimension
        x = tf.expand_dims(flatten(x), [0])

        # Now the LSTM layer
        cell = tf.contrib.rnn.LSTMCell(c.LSTM_UNITS, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, c.INPUT_KEEP_PROB,
                                             c.OUTPUT_KEEP_PROB)

        self.state_size = cell.state_size

        # Set up the initial state for episode resets later on
        self.c_init = np.zeros((1, cell.state_size.c), np.float32)
        self.h_init = np.zeros((1, cell.state_size.h), np.float32)

        # Need to hold the state of the LSTM for inputs from the previous frame
        self.c_in = tf.placeholder(tf.float32, [1, cell.state_size.c], name="c")
        self.h_in = tf.placeholder(tf.float32, [1, cell.state_size.h], name="h")
        state_in = rnn.LSTMStateTuple(self.c_in, self.h_in)

        outputs, (state_c, state_h) = tf.nn.dynamic_rnn(
            cell, x, initial_state=state_in,
            sequence_length=[tf.shape(self.x)[0]], time_major=False)

        # Make this a column vector to make the linear math easier
        x = tf.reshape(outputs, [-1, c.LSTM_UNITS])

        # As per A3C, logits and value function both are represented by linear
        # layers on top of the rest of the network
        self.logits = linear(x, action_size, "action",
                             normalized_columns_initializer(0.01))

        self.action = categorical_sample(self.logits, action_size)[0, :]

        # Also define the value function
        self.vf = tf.reshape(linear(x, 1, "value",
                                    normalized_columns_initializer(1.0)), [-1])

        # Other stuff
        self.state_out = [state_c[:1, :], state_h[:1, :]]

        # Training
        if c.HUMAN_TRAIN or c.ASYNC_HUMAN_TRAIN:
            # Training algo if human run is simple crossent
            self.labels = tf.placeholder(tf.float32, [None, c.NUM_ACTIONS])

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                        logits=self.logits))

            self.var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

            self.train = tf.train.AdamOptimizer(
                learning_rate=c.LEARNING_RATE).minimize(self.loss)

            # Saving op
            vars_to_save = self.var_list + [
                v for v in tf.global_variables() if v.name == 'global/global_step:0']
            self.saver = tf.train.Saver(vars_to_save)

            # Summary ops
            self.total_rew = tf.placeholder(tf.float32, name='totrew')
            bs = tf.to_float(tf.shape(self.x)[0])
            total_rew = tf.summary.scalar("model/total_reward", self.total_rew)
            si_sum = tf.summary.image("model/state", self.x)
            var_glob_sum = tf.summary.scalar("model/var_global_norm",
                                             tf.global_norm(self.var_list))
            loss_sum = tf.summary.scalar("model/loss",
                                         tf.reduce_mean(self.loss))
            self.summary_op = tf.summary.merge([si_sum, var_glob_sum,
                                                total_rew, loss_sum])

        elif scope != 'global':
            # TF graph for getting the gradients of the local model with A3C
            self.ac = tf.placeholder(tf.float32, [None, action_size],
                                     name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name='adv')
            self.r = tf.placeholder(tf.float32, [None], name='r')

            # A3C
            log_prob = tf.nn.log_softmax(self.logits)
            prob = tf.nn.softmax(self.logits)
            #   Minimize negative pi, or make adv big. Get prob of each 1hot
            #   action and multiply that action prob by the adv of the action
            pi_loss = -tf.reduce_sum(tf.reduce_sum(
                log_prob * self.ac, [1]) * self.adv)
            #   Squared difference between predicted values of states and actual
            #   rewards earned
            vf_loss = c.VF_LOSS_CONST * tf.reduce_sum(tf.square(
                self.vf - self.r))
            entropy = c.ENT_CONST * (-tf.reduce_sum(prob * log_prob))

            # Regularization
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              'worker{}'.format(worker_index))

            regularizer = c.REG_CONST * sum(
                tf.nn.l2_loss(x) for x in self.var_list)

            # Loss
            self.loss = pi_loss + vf_loss - entropy + regularizer

            grads = tf.gradients(self.loss, self.var_list)

            norm = None
            if c.MAX_GRAD_NORM:
                grads, norm = tf.clip_by_global_norm(grads, c.MAX_GRAD_NORM)

            # Train global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            'global')
            grads_and_vars = list(zip(grads, global_vars))

            self.adam_opt = tf.train.GradientDescentOptimizer(
                learning_rate=c.LEARNING_RATE)
            if c.ADAM:
                self.adam_opt = tf.train.AdamOptimizer(
                    learning_rate=c.LEARNING_RATE)

            self.train = self.adam_opt.apply_gradients(grads_and_vars)

            # Summary ops
            self.total_rew = tf.placeholder(tf.float32, name='totrew')
            bs = tf.to_float(tf.shape(self.x)[0])
            total_rew = tf.summary.scalar("model/total_reward", self.total_rew)
            pi_sum = tf.summary.scalar("model/policy_loss", pi_loss / bs)
            vf_sum = tf.summary.scalar("model/value_loss", vf_loss / bs)
            ent_sum = tf.summary.scalar("model/entropy", entropy / bs)
            si_sum = tf.summary.image("model/state", self.x)
            glob_sum = tf.summary.scalar("model/grad_global_norm",
                                         norm if norm is not None else tf.global_norm(
                                             grads))
            var_glob_sum = tf.summary.scalar("model/var_global_norm",
                                             tf.global_norm(global_vars))
            self.summary_op = tf.summary.merge([pi_sum, vf_sum, ent_sum, si_sum,
                                                glob_sum, var_glob_sum,
                                                total_rew])
        else:
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              'global')

            # Saving op
            vars_to_save = self.var_list + [
                v for v in tf.global_variables() if v.name == 'global/global_step:0']
            self.saver = tf.train.Saver(vars_to_save)

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
        if c.DEBUG and c.RANDOM_POLICY:
            # Move randomly
            x = random.randint(c.WINDOW_START[0],
                               c.WINDOW_END[0])
            y = random.randint(c.WINDOW_START[1],
                               c.WINDOW_END[1])
            click = random.randint(0, 1)

            return [[x, y, click], random.randint(0, 100), [None]]

        sess = tf.get_default_session()
        return sess.run([self.action, self.vf] + self.state_out,
                        {self.x: [ob], self.c_in: c_in, self.h_in: h_in})

    def value(self, ob, c_in, h_in):
        """
        Runs the value function.
        """
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.c_in: c_in,
                                  self.h_in: h_in})[0]

    def train_global(self, ob, ac, c_in, h_in, adv, reward, summary=False,
                     total_reward=0):
        """
        Calculates gradients from the reward based on A3C.
        Meant to be called from local networks.
        """
        sess = tf.get_default_session()
        outputs = [self.train]

        if summary:
            outputs = outputs + [self.summary_op]

        inputs = {
            self.x: ob,
            self.total_rew: total_reward,
            self.c_in: c_in,
            self.h_in: h_in
        }

        if c.HUMAN_TRAIN or c.ASYNC_HUMAN_TRAIN:
            logger.info("LABELS: %s", str(ac))
            inputs[self.labels] = ac
        else:
            inputs[self.adv] = adv
            inputs[self.ac] = ac
            inputs[self.r] = reward

        if c.MODEL_DEBUG:
            logger.info("INPUTS: %s", str(inputs))

        return sess.run(outputs, feed_dict=inputs)
