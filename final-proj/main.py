"""
Inits everything and runs the task.
"""

import tensorflow as tf

from a3c import A3C


sess = tf.Session()
with sess.as_default():
    # Init networks, tf vars, and everything else
    a3c = A3C()

    unitialized_vars = [var for var in tf.global_variables()
                        if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(unitialized_vars))

    # Kick off all of the worker threads that update the global graphs
    a3c.start_workers()

    # Update the global graphs based on experiences from the workers
    while True:
        a3c.process()

