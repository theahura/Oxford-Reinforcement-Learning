"""
Author: Amol Kapoor
Description: A3C Algorithm for async training on single multi-core CPU.
"""

import logging
import multiprocessing

import tensorflow as tf

import constants as c
from model import Policy
import runner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_model(session, env, scope):
    """
    Loads the tf model or inits a new one.
    """
    policy = Policy(env.observation_space.shape, c.NUM_ACTIONS, scope)

    ckpt = tf.train.get_checkpoint_state(c.CKPT_PATH)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("LOADING OLD MODEL")
        policy.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("LOADING NEW MODEL")
        session.run(tf.global_variables_initializer())
    return policy

class A3C(object):
    """
    Implementation of A3C. Inspired by starter agent.
    """

    def __init__(self, env, workers):
        """
        Set up global network sync graph.
        """
        self.env = env
        sess = tf.get_default_session()
        with tf.device('/cpu:0'):
            with tf.variable_scope("global"):
                self.global_network = get_model(sess, env, 'global')
                self.global_step = tf.get_variable(
                    "global_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)

        num_workers = workers if workers else multiprocessing.cpu_count()

        for i in range(num_workers):
            # Run worker threads
            #   in each worker thread call runner
            # Have a way to get relevant experiences from workers (shared queue?)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = get_model(sess, env, 'local')
                pi.global_step = self.global_step

            # Runner that interacts with the environment
            self.runner = RunnerThread(env, pi, 20, visualize)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for
                                   v1, v2 in zip(pi.var_list,
                                                 self.global_network.var_list)])

            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])
            self.local_steps = 0

    def start(self, sess):
        self.runner.start_runner(sess)

    def pull_batch_from_queue(self):
        """
        self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """

        # copy weights from shared to local
        sess.run(self.sync)

        # Get the latest experiences to process
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        # Debug every n steps
        should_compute_summary = self.task == 0 and (
            self.local_steps % c.DEBUG_STEPS == 0)

        # Get the gradients for the global network update from the local network
        # using the latest experiences
        fetched = self.local_network.get_gradients(batch.si, batch.features[0],
                                                   batch.features[1],
                                                   batch.adv, batch.r,
                                                   should_compute_summary)

        gradients = fetched[0]

        # Actually update the global network
        update = self.global_network.update_model(gradients)

        # Global and local networks have one more experience
        sess.run(self.global_step)
        self.local_steps += 1

        # Logs
        logger.info("Update: %s", str(update))
        if should_compute_summary:
            logger.info("Summary: %s", str(fetched[0]))
            logger.info("Gradients: %s", str(fetched[-1]))
