from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import DiagGaussianPdType
import numpy as np
from collections import namedtuple

# ob: observation without sysid added
# sysid: mass, inertia, etc...
# ob_concat: ob + sysid
# ac: action
# embed: dimensionality of embedding space
# agents: number of agents in batch environment
# window: length of rolling window for sysid network input
Dim = namedtuple('Dim', 'ob sysid ob_concat ac embed agents window')

class SysIDPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    # set up the network
    def _init(self, dim, hid_size, n_hid, gaussian_fixed_var=True):

        def MLPModule(last_out, n_hid, hid_size, last_initializer, n_out, name):
                for i in range(n_hid):
                    last_out = tf.nn.tanh(U.dense(last_out, hid_size,
                        name+"fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
                return U.dense(last_out, n_out,
                        name+"final", weight_init=U.normc_initializer(last_initializer))

        self.dim = dim
        self.alpha_sysid = 0.5

        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None, dim.ob_concat))
        with tf.variable_scope("ob_filter"):
            self.ob_rms = RunningMeanStd(shape=(dim.ob_concat))
            obz = tf.clip_by_value(
                (self.ob - self.ob_rms.mean) / self.ob_rms.std,
                -5.0, 5.0, name="ob_normalizer")
        obz, sysidz = tf.split(obz, [dim.ob, dim.sysid], axis=1)

        with tf.variable_scope("ob_white"):
            obz = tf.identity(obz)

        with tf.variable_scope("sysid_white"):
            sysidz = tf.identity(sysidz)

        with tf.variable_scope("embed"):
            self.embed = MLPModule(sysidz, n_hid, hid_size, 0.0, dim.embed, "embed")

        with tf.variable_scope("input_concat"):
            obz_and_embed = tf.concat([obz, self.embed], axis=1, name="input_concat")

        with tf.variable_scope("policy"):
            mean = MLPModule(obz_and_embed, n_hid, hid_size, 0.01, dim.ac, "pol")
            logstd = tf.get_variable(name="logstd", shape=[1, dim.ac], 
                initializer=tf.zeros_initializer())

        with tf.variable_scope("policy_to_gaussian"):
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
            self.pdtype = DiagGaussianPdType(dim.ac)
            self.pd = self.pdtype.pdfromflat(pdparam)

        with tf.variable_scope("value"):
            self.vpred = MLPModule(obz_and_embed, n_hid, hid_size, 1.0, 1, "vf")[:,0]

        self.traj_ob = U.get_placeholder(name="traj_ob",
            dtype=tf.float32, shape=[None, dim.window, dim.ob])
        self.traj_ac = U.get_placeholder(name="traj_ac",
            dtype=tf.float32, shape=[None, dim.window, dim.ac])
        with tf.variable_scope("sysid"):
            # SysID inputs, network, and loss function
            trajs_flat = tf.layers.flatten(tf.concat(
                [self.traj_ob, self.traj_ac], axis=2))
            self.traj2embed = MLPModule(trajs_flat,
                n_hid, hid_size, 1.0, dim.embed, "traj2embed")

        with tf.variable_scope("sysid_err_supervised"):
            self.sysid_err_supervised = tf.losses.mean_squared_error(
                tf.stop_gradient(self.embed), self.traj2embed)

        with tf.variable_scope("stochastic_switch"):
            self.stochastic = tf.placeholder(dtype=tf.bool, shape=(), name="stochastic")
            self.ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([self.stochastic, self.ob], [self.ac, self.vpred])

    # given the actual dynamics parameters, compute the embedding
    def sysid_to_embedded(self, sysid_vals):
        sz = sysid_vals.shape
        if len(sz) < 2:
            sysid_vals = sysid_vals.reshape((1, sz[0]))
        else:
            assert len(sz) == 2
        k = sysid_vals.shape[0]
        sysid_vals = np.concatenate([np.zeros((k, self.dim.ob)), sysid_vals], axis=1)
        sess = tf.get_default_session()
        embed = sess.run(self.embed, feed_dict={self.ob: sysid_vals})
        if len(sz) < 2:
            return np.squeeze(embed)
        else:
            return embed

    # given the ob/ac windows, estimate the embedding
    def estimate_sysid(self, ob_trajs, ac_trajs):
        feed = {
            self.traj_ob : ob_trajs,
            self.traj_ac : ac_trajs,
        }
        sess = tf.get_default_session()
        embed = sess.run(self.traj2embed, feed_dict=feed)
        return embed

    # act - ob is concat(ob, sysid)
    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob)
        return ac1, vpred1

    # act, but with a given embedding value supplanting
    # the embedding computed from the true sysid values
    def act_embed(self, stochastic, ob, embed):
        feed = {
            self.ob : ob,
            self.embed : embed,
            self.stochastic : stochastic,
        }
        sess = tf.get_default_session()
        ac = sess.run(self.ac, feed_dict=feed)
        return ac

    # for OpenAI Baselines compatibility
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

