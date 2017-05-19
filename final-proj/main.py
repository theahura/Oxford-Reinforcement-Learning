"""
Inits everything and runs the task.
"""

import tensorflow as tf

from a3c import A3C


sess = tf.Session()
init = tf.global_variables_initializer()
with sess.as_default():
    sess.run(init)
    # Init networks and everything else
    a3c = A3C()

    # Kick off all of the worker threads that update the global graphs
    a3c.start_workers()

    # Update the global graphs based on experiences from the workers
    while True:
        a3c.process()

