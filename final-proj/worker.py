"""
Author: Amol Kapoor
Description: Where the environment is run and experiences are generated.
"""

import threading
import time
import logging

import constants as c
import envs
import model
import runner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Worker(threading.Thread):
    """
    Run the thread to interact with the environment.
    """

    def __init__(self, sess, global_q, worker_index):

        if c.DEBUG:
            logger.info('WORKER %d STARTED.', worker_index)
        threading.Thread.__init__(self)
        self.daemon = True
        self.worker_index = worker_index
        self.q = global_q
        self.sess = sess
        self.policy = model.get_model(sess, 'local', worker_index)
        self.local_steps = 0
        self.is_running = True

    def start_runner(self):
        """
        Wrapper for start.
        """
        self.start()

    def run(self):
        """
        Overrides previous run method to init tf.
        """
        while True:
            with self.sess.as_default():
                try:
                    self._run()
                except:
                    logger.info("WORKER %d DIED", self.worker_index)
                    self.is_running = False
                    time.sleep(c.SLEEP_TIME)
                    self.is_running = True
                    logger.info("RESTARTING WORKER %d", self.worker_index)

    def _run(self):
        """
        Main worker loop. Loads experiences into the global q.
        """

        self.env = envs.create_env()
        rollout_provider = runner.run_env(self.env, self.policy,
                                          self.worker_index)
        while True:
            self.local_steps += 1
            self.q.put(next(rollout_provider))
            logger.info("STEPS IN WORKER %d: %d", self.worker_index,
                        self.local_steps)
