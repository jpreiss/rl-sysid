from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

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

    def _init(self, ob_space, ac_space, sysid_dim, latent_dim, hid_size, n_hid, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        assert isinstance(ac_space, gym.spaces.Box)
        assert len(ob_space.shape) == 1
        assert len(ac_space.shape) == 1
        ob_dim = ob_space.shape[0] - sysid_dim
        ac_dim = ac_space.shape[0]

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz, sysidz = tf.split(obz, [ob_dim, sysid_dim], axis=1)

        self.embed = MLPModule(sysidz, n_hid, hid_size, 0.1, latent_dim, "embed")

        obz_and_embed = tf.concat([obz, self.embed], axis=1)

        self.vpred = MLPModule(obz_and_embed, n_hid, hid_size, 1.0, 1, "vf")[:,0]

        mean = MLPModule(obz_and_embed, n_hid, hid_size, 0.01, ac_dim, "pol")
        logstd = tf.get_variable(name="logstd", shape=[1, ac_dim], 
            initializer=tf.zeros_initializer())
        pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob)
        return ac1, vpred1
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

