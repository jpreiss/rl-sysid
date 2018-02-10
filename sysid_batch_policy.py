from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
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

# dummy policy that ignores trajectory input
class SysIDPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    # set up the network
    def _init(self, dim, hid_size, n_hid, use_embedding=True, alpha_sysid=1):

        def MLPModule(last_out, n_hid, hid_size, last_initializer, n_out, name, middle_initializer=1.0):
                for i in range(n_hid):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size,
                        name=name+"fc%i"%(i+1), kernel_initializer=U.normc_initializer(middle_initializer)))
                return tf.layers.dense(last_out, n_out,
                        name=name+"final", kernel_initializer=U.normc_initializer(last_initializer))

        self.dim = dim
        self.alpha_sysid = alpha_sysid

        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None, dim.ob_concat))
        self.traj_ob = U.get_placeholder(name="traj_ob",
            dtype=tf.float32, shape=[None, dim.window, dim.ob])
        self.traj_ac = U.get_placeholder(name="traj_ac",
            dtype=tf.float32, shape=[None, dim.window, dim.ac])

        ob, sysid = tf.split(self.ob, [dim.ob, dim.sysid], axis=1)
        with tf.variable_scope("ob_filter"):
            self.ob_rms = RunningMeanStd(shape=(dim.ob_concat))
            obz_all = tf.clip_by_value(
                (self.ob - self.ob_rms.mean) / self.ob_rms.std,
                -5.0, 5.0, name="ob_normalizer")
        obz, sysidz = tf.split(obz_all, [dim.ob, dim.sysid], axis=1)

        with tf.variable_scope("ob_white"):
            obz = tf.identity(obz)

        with tf.variable_scope("sysid_white"):
            sysidz = tf.identity(sysidz)

        with tf.variable_scope("pol"):
            with tf.variable_scope("sysid_preprocessor"):
                if use_embed:
                    self.embed = MLPModule(sysidz, n_hid, hid_size, 0.0, dim.embed, "embed")
                else:
                    self.embed = tf.identity(sysidz)

            with tf.variable_scope("input_concat"):
                obz_and_embed = tf.concat([obz, self.embed], axis=1, name="input_concat")

            with tf.variable_scope("policy"):
                mean = MLPModule(obz_all, n_hid, hid_size, 0.01, dim.ac, "pol")
                logstd = tf.get_variable(name="logstd", shape=[1, dim.ac], 
                    initializer=tf.constant_initializer(0))

            with tf.variable_scope("policy_to_gaussian"):
                pdparam = tf.concat([mean, mean * 0.0 + logstd], 1)
                self.pdtype = DiagGaussianPdType(dim.ac)
                self.pd = self.pdtype.pdfromflat(pdparam)

        with tf.variable_scope("vf"):
            self.vpred = MLPModule(obz_all, n_hid, hid_size, 0.1, 1, "vf")[:,0]

        with tf.variable_scope("sysid"):
            trajs = tf.concat([self.traj_ob, self.traj_ac], axis=2)
            self.traj2embed = sysid_convnet(trajs, dim.sysid)

        with tf.variable_scope("sysid_err_supervised"):
            self.sysid_err_supervised = tf.losses.mean_squared_error(
                tf.stop_gradient(self.embed), self.traj2embed)

        with tf.variable_scope("stochastic_switch"):
            self.stochastic = tf.placeholder(dtype=tf.bool, shape=(), name="stochastic")
            self.ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([self.stochastic, self.ob], [self.ac, self.vpred])

    # given the actual dynamics parameters, compute the embedding
    def sysid_to_embedded(self, sysid_vals):
        # pass val[None,:] if needing to evaluate for just one sysid val
        assert len(sysid_vals.shape) == 2
        k = sysid_vals.shape[0]
        sysid_vals = np.concatenate([np.zeros((k, self.dim.ob)), sysid_vals], axis=1)
        sess = tf.get_default_session()
        embed = sess.run(self.embed, feed_dict={self.ob: sysid_vals})
        return embed

    # given the ob/ac windows, estimate the embedding
    def estimate_sysid(self, ob_trajs, ac_trajs):
        feed = {
            self.traj_ob : ob_trajs,
            self.traj_ac : ac_trajs,
        }
        sess = tf.get_default_session()
        embed = sess.run(self.traj2sysid, feed_dict=feed)
        return embed

    # act - ob is concat(ob, sysid)
    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob)
        return ac1, vpred1

    # for OpenAI Baselines compatibility
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

