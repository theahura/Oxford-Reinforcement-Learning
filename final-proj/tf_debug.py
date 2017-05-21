"""
Used to compare the weights from two tf checkpoints.
"""
import tensorflow as tf

from model import Policy
import constants as c

ckpt_num = raw_input("ckpt num: ")
CKPT_PATH = './data/ckpt/slither.ckpt-%d.meta' % int(ckpt_num)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    with tf.variable_scope('global'):
        print 'getting policy'
        policy = Policy(c.OBSERVATION_SPACE, c.NUM_ACTIONS, 'global')

        print 'getting ckpt'
        print CKPT_PATH

        saver = tf.train.import_meta_graph(CKPT_PATH)
        saver.restore(sess, tf.train.latest_checkpoint('./data/ckpt/'))

        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='global')
        tvars_vals = sess.run(tvars)

        for var, val in zip(tvars, tvars_vals):
            print(var, val)
