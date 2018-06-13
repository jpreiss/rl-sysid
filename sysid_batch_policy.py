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

# construct a MLP policy with tanh activation in middle layers
def MLPModule(np_random, last_out, n_hid, hid_size, last_initializer, n_out, name, middle_initializer=1.0):
    for i in range(n_hid):
        last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size,
            name=name+"fc%i"%(i+1), kernel_initializer=U.normc_initializer(np_random, middle_initializer)))
    return tf.layers.dense(last_out, n_out,
            name=name+"final", kernel_initializer=U.normc_initializer(np_random, last_initializer))

# construct the 1D convolutional network for (ob, ac)^k -> SysID
def sysid_convnet(np_random, input, sysid_dim):
    window = input.shape[1]

    if window >= 20: # old
        conv1 = tf.layers.conv1d(input, filters=32, kernel_size=3, activation=None, kernel_initializer=U.normc_initializer(np_random, 0.1))
        conv2 = tf.layers.conv1d(conv1, filters=32, kernel_size=3, activation=tf.nn.relu, kernel_initializer=U.normc_initializer(np_random, 0.1))
        pool = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2)

        conv4 = tf.layers.conv1d(pool, filters=32, kernel_size=5, activation=tf.nn.relu, kernel_initializer=U.normc_initializer(np_random, 0.1))
        flat = tf.layers.flatten(conv4)

    else:
        conv1 = tf.layers.conv1d(input, filters=64, kernel_size=3, activation=None, kernel_initializer=U.normc_initializer(np_random, 0.1))
        conv2 = tf.layers.conv1d(conv1, filters=64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=U.normc_initializer(np_random, 0.1))
        conv3 = tf.layers.conv1d(conv2, filters=64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=U.normc_initializer(np_random, 0.1))
        flat = tf.layers.flatten(conv3)
        flat2 = tf.layers.dense(flat, 128, kernel_initializer=U.normc_initializer(np_random, 0.1), activation=tf.nn.relu)
        #print("convnet flat size:", flat.shape)

    fc = tf.layers.dense(flat2, sysid_dim, kernel_initializer=U.normc_initializer(np_random, 0.1))
    return fc

# policy flavors:
BLIND = "blind" # no access to SysID values - domain randomization only
PLAIN = "plain" # MLP, access to true SysID values concatenated w/ obs
EXTRA = "extra" # Access to true SysID w/ extra SysID processing network
EMBED = "embed" # Converts SysID to embedding ("ours")
TRAJ  = "traj"  # acces to trajectory history wired directly to policy, no supervised part

# TODO option to make TRPO pass in the trajectory at train time
# so the TRAJ flavor can work
flavors = [BLIND, EXTRA, EMBED] #, TRAJ]
#flavors = [PLAIN]

flavor_uses_sysid = {
    BLIND: False,
    PLAIN: True,
    EXTRA: True,
    EMBED: True,
    TRAJ:  False,
}
for flavor in flavors:
    assert flavor in flavor_uses_sysid

class SysIDPolicy(object):

    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    # set up the network
    # NOTE: due to normalization of SysID values and KL-regularization of embedding space,
    # alpha_sysid shouldn't need to vary between environments - but we'll see...
    def _init(self, np_random, flavor, dim, hid_size=32, n_hid=2, alpha_sysid=0.1, test=False):

        print("obs dim:", dim.ob)

        # inputs & hyperparameters
        self.flavor = flavor
        self.dim = dim
        self.alpha_sysid = alpha_sysid
        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None, dim.ob_concat))
        self.ob_traj = U.get_placeholder(name="ob_traj",
            dtype=tf.float32, shape=[None, dim.window, dim.ob])
        self.ac_traj = U.get_placeholder(name="ac_traj",
            dtype=tf.float32, shape=[None, dim.window, dim.ac])

        # regular inputs whitening
        ob, sysid = tf.split(self.ob, [dim.ob, dim.sysid], axis=1)
        with tf.variable_scope("ob_filter"):
            self.ob_rms = RunningMeanStd(shape=(dim.ob_concat))
            obz_all = tf.clip_by_value(
                (self.ob - self.ob_rms.mean) / self.ob_rms.std,
                -5.0, 5.0, name="ob_normalizer")
        obz, sysidz = tf.split(obz_all, [dim.ob, dim.sysid], axis=1)
        print("obz dim:", obz.shape, "sysidz dim:", sysidz.shape)
        with tf.variable_scope("ob_white"):
            obz = tf.identity(obz)
        with tf.variable_scope("sysid_white"):
            self.sysidz = tf.identity(sysidz)

        # trajectory inputs for SysID
        # NOTE: the environment should be defined such that
        # actions are relatively close to Normal(0,1)
        ob_trajz = tf.clip_by_value(
            (self.ob_traj - self.ob_rms.mean[:dim.ob]) / self.ob_rms.std[:dim.ob],
            -5.0, 5.0, name="ob_traj_white")
        trajs = tf.concat([ob_trajz, self.ac_traj], axis=2)

        # these rewards will be optimized via direct gradient-based optimization
        # (not RL reward), in the same place as e.g. the entropy regularization
        self.extra_rewards = []
        self.extra_reward_names = []

        with tf.variable_scope("sysid"):
            if flavor == PLAIN:
                self.traj2sysid = sysid_convnet(np_random, trajs, dim.sysid)
            elif flavor == EXTRA:
                self.traj2sysid = sysid_convnet(np_random, trajs, dim.sysid)
            elif flavor == EMBED:
                self.traj2embed = sysid_convnet(np_random, trajs, dim.embed)

        EMBED_N_HID = 2
        EMBED_HID_SZ = 2 * dim.sysid

        # policy
        with tf.variable_scope("pol"):
            if flavor == BLIND:
                policy_input = obz
                self.sysid_err_supervised = tf.constant(0.0)
            elif flavor == PLAIN:
                self.sysid_err_supervised = tf.losses.mean_squared_error(
                    tf.stop_gradient(sysidz), self.traj2sysid)
                policy_input = tf.concat([obz, self.traj2sysid]) if test else obz_all
            elif flavor == EXTRA:
                sysid_processor_input = self.traj2sysid if test else sysidz
                sysid_processor = MLPModule(np_random, sysid_processor_input,
                    EMBED_N_HID, EMBED_HID_SZ, 1.0, dim.embed, "sysid_processor")
                policy_input = tf.concat([obz, sysid_processor], axis=1, name="input_concat")
                self.sysid_err_supervised = tf.losses.mean_squared_error(
                    tf.stop_gradient(sysidz), self.traj2sysid)
            elif flavor == EMBED:
                self.embed = MLPModule(np_random, sysidz,
                    EMBED_N_HID, EMBED_HID_SZ, 1.0, dim.embed, "embed")
                embed_input = self.traj2embed if test else self.embed
                policy_input = tf.concat([obz, embed_input], axis=1, name="input_concat")
                self.sysid_err_supervised = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.embed), self.traj2embed)
                mean, var = tf.nn.moments(self.embed, 0)
                dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))
                std_dist = tf.distributions.Normal(loc=0.0, scale=1.0)
                embed_KL = tf.reduce_mean(tf.distributions.kl_divergence(dist, std_dist))
                self.extra_rewards.append(-0.1 * embed_KL)
                self.extra_reward_names.append("neg_embed_KL")
            elif flavor == TRAJ:
                self.traj_conv = sysid_convnet(np_random, trajs, dim.embed)
                policy_input = tf.concat([obz, self.traj_conv], axis=1, name="input_concat")
                self.sysid_err_supervised = tf.constant(0.0)
            else:
                raise ValueError("flavor '{}' does not exist".format(flavor))

            # main policy MLP. outputs mean and logstd of stochastic Gaussian policy
            with tf.variable_scope("policy"):
                print("policy input dimensionality:", policy_input.get_shape().as_list())
                mean = MLPModule(np_random, policy_input, n_hid, hid_size, 0.01, dim.ac, "pol")
                logstd = tf.maximum(tf.get_variable(name="logstd", shape=[1, dim.ac], 
                    initializer=tf.constant_initializer(-0.3)), -1.0)

            with tf.variable_scope("policy_to_gaussian"):
                pdparam = tf.concat([mean, mean * 0.0 + logstd], 1)
                self.pdtype = DiagGaussianPdType(dim.ac)
                self.pd = self.pdtype.pdfromflat(pdparam)

        # value function
        with tf.variable_scope("vf"):
            self.vpred = MLPModule(np_random, tf.stop_gradient(policy_input),
                n_hid, hid_size, 0.1, 1, "vf")[:,0]

        # switch between stochastic and deterministic policy
        with tf.variable_scope("stochastic_switch"):
            self.stochastic = tf.placeholder(dtype=tf.bool, shape=(), name="stochastic")
            self.ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())

        # function we'll call when interacting with environment
        self._act = U.function([self.stochastic, self.ob], [self.ac, self.vpred])

        # for test time, the trajectory is fed in
        self._act_traj = U.function(
            [self.stochastic, self.ob, self.ob_traj, self.ac_traj],
            [self.ac, self.vpred])


    # given the actual dynamics parameters, compute the embedding
    def sysid_to_embedded(self, sysid_vals):
        if self.flavor in [BLIND, TRAJ]:
            # could also just return sysid_vals, but this draws attention to lack of sysid
            return 0 * sysid_vals

        # pass val[None,:] if needing to evaluate for just one sysid val
        assert len(sysid_vals.shape) == 2
        k = sysid_vals.shape[0]
        sysid_vals = np.concatenate([np.zeros((k, self.dim.ob)), sysid_vals], axis=1)
        sess = tf.get_default_session()

        if self.flavor == EMBED:
            embed = sess.run(self.embed, feed_dict={self.ob: sysid_vals})
            return embed
        else:
            sysidz = sess.run(self.sysidz, feed_dict={self.ob: sysid_vals})
            return sysidz

    # given the ob/ac trajectories, estimate the embedding.
    # it's also part of the main policy, but needed on its own for TRPO.
    def estimate_sysid(self, ob_trajs, ac_trajs):
        sess = tf.get_default_session()
        N = ob_trajs.shape[0]
        k = N // 2048 + 1

        if self.flavor in [BLIND, TRAJ]:
            return np.zeros((N, self.dim.sysid))

        # TODO use tf.data or something to do this automatically!
        def gen(ob_splits, ac_splits):
            for o, a in zip(ob_splits, ac_splits):
                feed = {
                    self.ob_traj : o,
                    self.ac_traj : a,
                }
                if self.flavor == EMBED:
                    yield sess.run(self.traj2embed, feed_dict=feed)
                else:
                    yield sess.run(self.traj2sysid, feed_dict=feed)
        est = np.vstack(gen(
            np.array_split(ob_trajs, k), np.array_split(ac_trajs, k)))
        return est

    # act - ob is concat(ob, sysid)
    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob)
        return ac1, vpred1

    def act_traj(self, stochastic, ob, ob_traj, ac_traj):
        return self._act_traj(stochastic, ob, ob_traj, ac_traj)

    # for OpenAI Baselines compatibility
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

