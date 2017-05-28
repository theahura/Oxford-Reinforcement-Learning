"""
Author: Amol Kapoor
Description: A3C Algorithm for async training on single multi-core CPU.
"""

import logging
import multiprocessing
import os
import Queue

import numpy as np
import scipy.signal
import tensorflow as tf

import constants as c
import envs
import humantest
import model
import runner
from worker import Worker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def discount(x, gamma):
    """
    Discounts rewards.
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
        'worker': rollout.worker,
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
        self.global_q = Queue.Queue()
        sess = tf.get_default_session()
        with tf.device('/cpu:0'):
            with tf.variable_scope('global'):
                self.global_steps = tf.Variable(0, name='global_step',
                                                trainable=False)
                self.policy = model.get_model(sess, 'global')
                self.glob_inc = tf.assign(self.global_steps,
                                          self.global_steps+1)

            num_workers = multiprocessing.cpu_count() - 1
            if c.NUM_WORKERS:
                num_workers = c.NUM_WORKERS - 1

            self.workers = []
            for i in range(num_workers):
                with tf.variable_scope('worker{}'.format(i)):
                    # Create a new worker thread
                    worker = Worker(sess, self.global_q, i, c.ASYNC_HUMAN_TRAIN)

                    # And sync it to the global thread
                    worker.sync(sess, self.policy.var_list)
                    self.workers.append(worker)

        self.summary_writer = tf.summary.FileWriter(c.LOGDIR)

    def start_workers(self):
        """
        Kicks off all the worker threads after init.
        """
        for worker in self.workers:
            worker.start_runner()

    def pull_batch_from_queue(self):
        """
        Gets all of the experiences available on call
        """
        if c.GLOBAL_DEBUG:
            logger.info("GLOBAL Q LEN: %d", self.global_q.qsize())
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

        if c.GLOBAL_DEBUG:
            logger.info("GLOBAL PROCESS STARTED")

        sess = tf.get_default_session()

        # Get the latest experiences to process
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout)

        if c.GLOBAL_DEBUG:
            logger.info("GLOBAL BATCH RECEIVED: %d",
                        sess.run(self.global_steps))

        # Debug every n steps
        should_compute_summary = sess.run(self.global_steps) % c.SUM_STEPS == 0

        if c.GLOBAL_DEBUG:
            logger.info("BATCH: %s", str(batch))
            logger.info("WORKER INDEX: %d", batch['worker'])
            logger.info("TOTAL REWARD: %d", rollout.total_reward)

        if c.GLOBAL_DEBUG and not c.HUMAN_TRAIN:
            for w in self.workers:
                logger.info("WORKER %d RUNNING: %s", w.worker_index,
                            w.is_running)

        worker = self.workers[batch['worker']]
        # Update the global network from the local workers' gradients
        fetched = worker.policy.train_global(batch['si'], batch['a'],
                                             batch['features'][0],
                                             batch['features'][1], batch['adv'],
                                             batch['r'], should_compute_summary,
                                             rollout.total_reward)

        if c.GLOBAL_DEBUG:
            logger.info("GLOBAL UPDATE DONE")

        # Copy the changes back down to the local network
        if not c.HUMAN_TRAIN:
            worker.sync(sess, self.policy.var_list)

        if c.GLOBAL_DEBUG:
            logger.info("GLOBAL SYNC FINISHED")

        # Global network has one more experience
        sess.run(self.glob_inc)

        if sess.run(self.global_steps) % c.STEPS_TO_SAVE == 0 or c.HUMAN_TRAIN:
            logger.info("SAVING %d", sess.run(self.global_steps))
            checkpoint_path = os.path.join(c.CKPT_PATH, 'slither.ckpt')
            self.policy.saver.save(sess, checkpoint_path,
                                   global_step=self.global_steps)

        if sess.run(self.global_steps) % c.LEARNING_RATE_STEP == 0:
            c.LEARNING_RATE = c.LEARNING_RATE * c.LEARNING_RATE_SCALE

        # Logs
        if should_compute_summary:
            logger.info("GETTING THE SUMMARY")
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[1]),
                                            sess.run(self.global_steps))

            self.summary_writer.flush()

    def play(self):
        """
        Play instead of train.
        """
        if c.GLOBAL_DEBUG:
            logger.info("PLAYING GAME")
        env = envs.create_env()
        rollout_provider = runner.run_env(env, self.policy, 0)
        steps = 0
        while True:
            steps += 1
            print process_rollout(next(rollout_provider))

    def humantrain(self):
        """
        Over the shoulder learning.
        """
        sess = tf.get_default_session()
        logger.info("RUNNING LOCAL HUMAN STEP %d", sess.run(self.global_steps))
        self.workers.append(self)
        try:
            humantest.setup_keyboard()
            env = envs.create_env()
            rollout_provider = runner.run_env(env, self.policy, 0, True)
            steps = 0
            while True:
                steps += 1
                self.global_q.put(next(rollout_provider))
                self.process()
                logger.info("STEPS IN WORKER %d: %d", 0, steps)
                raise ValueError
        except Exception as e:
            logger.info("ERROR: %s", str(e))
            humantest.return_keyboard()
            quit()
        finally:
            humantest.return_keyboard()
            quit()
