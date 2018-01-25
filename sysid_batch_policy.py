from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

def MLPModule(last_out, n_hid, hid_size, last_initializer, n_out, name):
        for i in range(n_hid):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size,
                name+"fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        return U.dense(last_out, n_out,
                name+"final", weight_init=U.normc_initializer(last_initializer))

class SysIDPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, sysid_dim, latent_dim, traj_len, hid_size, n_hid, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        assert isinstance(ac_space, gym.spaces.Box)
        assert len(ob_space.shape) == 1
        assert len(ac_space.shape) == 1
        ob_dim = ob_space.shape[0] - sysid_dim
        ac_dim = ac_space.shape[0]

        self.ob_dim = ob_dim
        self.traj_len = traj_len
        self.latent_dim = latent_dim
        self.pdtype = pdtype = make_pdtype(ac_space)
        self.alpha_sysid = 1.0
        sequence_length = None

        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz, sysidz = tf.split(obz, [ob_dim, sysid_dim], axis=1)

        self.embed = MLPModule(sysidz, n_hid, hid_size, 0.0, latent_dim, "embed")

        obz_and_embed = tf.concat([obz, self.embed], axis=1)

        #self.vpred = MLPModule(obz_and_embed, n_hid, hid_size, 1.0, 1, "vf")[:,0]
        self.vpred = MLPModule(obz, n_hid, hid_size, 1.0, 1, "vf")[:,0]

        mean = MLPModule(obz_and_embed, n_hid, hid_size, 0.01, ac_dim, "pol")
        logstd = tf.get_variable(name="logstd", shape=[1, ac_dim], 
            initializer=tf.zeros_initializer())
        pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        self.pd = pdtype.pdfromflat(pdparam)

        # SysID inputs, network, and loss function
        self.traj_ob = U.get_placeholder(name="traj_ob",
            dtype=tf.float32, shape=[None, traj_len, ob_dim])
        self.traj_ac = U.get_placeholder(name="traj_ac",
            dtype=tf.float32, shape=[None, traj_len, ac_dim])
        trajs_flat = tf.layers.flatten(tf.concat(
            [self.traj_ob, self.traj_ac], axis=2))
        self.traj2embed = MLPModule(trajs_flat,
            n_hid, hid_size, 1.0, latent_dim, "traj2embed")
        self.sysid_err_supervised = tf.losses.mean_squared_error(
            tf.stop_gradient(self.embed), self.traj2embed)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, self.ob], [ac, self.vpred])

    def sysid_to_embedded(self, sysid_vals):
        sz = sysid_vals.shape
        if len(sz) < 2:
            sysid_vals = sysid_vals.reshape((1, sz[0]))
        else:
            assert len(sz) == 2
        k = sysid_vals.shape[0]
        sysid_vals = np.concatenate([np.zeros((k, self.ob_dim)), sysid_vals], axis=1)
        sess = tf.get_default_session()
        embed = sess.run(self.embed, feed_dict={self.ob: sysid_vals})
        return np.squeeze(embed)

    def estimate_sysid(self, ob_trajs, ac_trajs):
        feed = {
            self.traj_ob : ob_trajs,
            self.traj_ac : ac_trajs,
        }
        sess = tf.get_default_session()
        embed = sess.run(self.traj2embed, feed_dict=feed)
        return embed

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob)
        return ac1, vpred1
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

