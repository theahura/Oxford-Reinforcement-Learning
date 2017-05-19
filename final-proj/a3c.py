"""
Author: Amol Kapoor
Description: A3C Algorithm for async training on single multi-core CPU.
"""

import logging
import multiprocessing
import os
import Queue

import numpy as np
import scipy
import tensorflow as tf

import constants as c
import model
from worker import Worker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def discount(x, gamma):
    """
    Discounts x. Part of adv calculation. See starter agent.
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout):
    """
    Given a rollout, get returns and advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, c.GAMMA)[:-1]
    delta_t = rewards + c.GAMMA * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, c.GAMMA * c.LAMBDA)

    features = rollout.features[0]
    batch = {
        'si': batch_si,
        'a': batch_a,
        'adv': batch_adv,
        'r': batch_r,
        'terminal': rollout.terminal,
        'features': features,
        'worker': rollout.worker
    }
    return batch

class A3C(object):
    """
    Implementation of A3C. Inspired by starter agent.
    """

    def __init__(self):
        """
        Set up global network sync graph.
        """
        self.global_steps = 0
        self.global_q = Queue.Queue()
        sess = tf.get_default_session()
        with tf.device('/cpu:0'):
            with tf.variable_scope('global'):
                self.global_network = model.get_model(sess, 'global')

            num_workers = multiprocessing.cpu_count() - 1
            if c.NUM_WORKERS:
                num_workers = c.NUM_WORKERS - 1

            self.workers = []
            for i in range(num_workers):
                with tf.variable_scope('worker{}'.format(i)):
                    self.workers.append(Worker(sess, self.global_q, i))

    def start_workers(self):
        for worker in self.workers:
            worker.start_runner()

    def pull_batch_from_queue(self):
        """
        Gets all of the experiences available on call
        """
        rollout = self.global_q.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.global_q.get_nowait())
            except Queue.Empty:
                break
        return rollout

    def process(self):
        """
        process grabs a rollout that's been produced by a thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        sess = tf.get_default_session()

        # Get the latest experiences to process
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout)

        # Debug every n steps
        should_compute_summary = self.global_steps % c.DEBUG_STEPS == 0

        worker = self.workers[batch['worker']]

        # Get the gradients for the global network update from the local network
        # using the latest experiences
        fetched = worker.policy.get_gradients(batch['si'], batch['a'],
                                              batch['features'][0],
                                              batch['features'][1],
                                              batch['adv'], batch['r'],
                                              should_compute_summary)

        gradients = fetched[0]

        # Actually update the global network
        update = self.global_network.update_model(gradients)

        # Copy the changes back down to the local network
        sync = tf.group(*[v1.assign(v2) for v1, v2 in
                          zip(worker.policy.var_list,
                              self.global_network.var_list)])
        sess.run(sync)

        # Global network has one more experience
        self.global_steps += 1

        if self.global_steps % c.STEPS_TO_SAVE == 0:
            checkpoint_path = os.path.join(c.CKPT_PATH, 'slither.ckpt')
            self.global_network.saver.save(sess, checkpoint_path,
                                           global_step=self.global_steps)

        # Logs
        logger.info("Update step: %d, Update: %s", self.global_steps,
                    str(update))
        if should_compute_summary:
            logger.info("Summary: %s", str(fetched[0]))
            logger.info("Gradients: %s", str(fetched[-1]))
