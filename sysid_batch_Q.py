from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

# ob: observation without sysid added
# sysid: mass, inertia, etc...
# ob_concat: ob + sysid
# ac: action
# embed: dimensionality of embedding space
# agents: number of agents in batch environment
# window: length of rolling window for sysid network input
Dim = namedtuple('Dim', 'ob sysid ob_concat ac embed agents window')

# construct a MLP policy with tanh activation in middle layers
def MLPModule(last_out, n_hid, hid_size, n_out, name):
    for i in range(n_hid):
        last_out = tf.layers.dense(last_out, hid_size,
            name=f"{name}_fc{i+1}", activation=tf.nn.relu)
    return tf.layers.dense(last_out, n_out, name=name+"final")

# construct the 1D convolutional network for (ob, ac)^k -> SysID
def sysid_convnet(input, sysid_dim):
    conv1 = tf.layers.conv1d(input, filters=64, kernel_size=3, activation=None)
    conv2 = tf.layers.conv1d(conv1, filters=64, kernel_size=3, activation=tf.nn.relu)
    conv3 = tf.layers.conv1d(conv2, filters=64, kernel_size=3, activation=tf.nn.relu)
    flat = tf.layers.flatten(conv3)
    print("convnet flat size:", flat.shape)
    flat2 = tf.layers.dense(flat, 128, activation=tf.nn.relu)
    fc = tf.layers.dense(flat2, sysid_dim)
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

class BufferSampler(object):

    def __init__(self, samples):
        self.samples = samples.flatten()
        self.N = samples.size
        self.shuffle = np.arange(self.N)
        self.cursor = 0

    def sample(self, npr, shape):
        sz = np.prod(shape)
        assert sz < self.N // 10
        if self.cursor + sz > self.N:
            npr.shuffle(self.shuffle)
            self.cursor = 0
        end = self.cursor + sz
        x = self.samples[self.cursor:end].reshape(shape)
        self.cursor = end
        return x



class SysIDQ(object):

    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    # set up the network
    # NOTE: due to normalization of SysID values and KL-regularization of embedding space,
    # alpha_sysid shouldn't need to vary between environments - but we'll see...
    def _init(
        self, flavor, dim,
        hid_size=32, n_hid=2,
        alpha_sysid=0.1, test=False,
        cem_count=128, cem_keep=16, cem_iters=4, cem_smooth=0.7
        ):

        # amazingly (or not?), if we compute fresh N(0,1) samples every time,
        # it's a huge computational bottleneck. Use a cache instead.
        # TODO: pass in npr
        self.npr = np.random.RandomState()
        self.normal_sampler = BufferSampler(self.npr.normal(size=int(1e5)))
        self.uniform_sampler = BufferSampler(self.npr.uniform(-1.0, 1.0, size=int(1e5)))


        # inputs & hyperparameters
        self.flavor = flavor
        self.dim = dim
        self.alpha_sysid = alpha_sysid
        self.cem_count = cem_count
        self.cem_keep = cem_keep
        self.cem_iters = cem_iters
        self.cem_smooth = cem_smooth

        self.in_ob = tf.placeholder(tf.float32, shape=(None, dim.ob_concat), name="ob")
        self.in_ac = tf.placeholder(tf.float32, shape=(None, dim.ac), name="ac")
        self.in_ob_traj = tf.placeholder(tf.float32,
            shape=(None, dim.window, dim.ob), name="ob_traj")
        self.in_ac_traj = tf.placeholder(tf.float32,
            shape=(None, dim.window, dim.ac), name="ac_traj")

        self.is_train = tf.Variable(not test, name="is_train")

        # regular inputs whitening
        ob, sysid = tf.split(self.in_ob, [dim.ob, dim.sysid], axis=1)
        with tf.variable_scope("ob_filter"):
            self.ob_rms = RunningMeanStd(shape=(dim.ob_concat))
            obz_all = tf.clip_by_value(
                (self.in_ob - self.ob_rms.mean) / self.ob_rms.std,
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
            (self.in_ob_traj - self.ob_rms.mean[:dim.ob]) / self.ob_rms.std[:dim.ob],
            -5.0, 5.0, name="ob_traj_white")
        trajs = tf.concat([ob_trajz, self.in_ac_traj], axis=2)

        # these rewards will be optimized via direct gradient-based optimization
        # (not RL reward), in the same place as e.g. the entropy regularization
        self.extra_rewards = []
        self.extra_reward_names = []

        with tf.variable_scope("sysid"):
            if flavor == PLAIN:
                self.traj2sysid = sysid_convnet(trajs, dim.sysid)
            elif flavor == EXTRA:
                self.traj2sysid = sysid_convnet(trajs, dim.sysid)
            elif flavor == EMBED:
                self.traj2embed = sysid_convnet(trajs, dim.embed)

        EMBED_N_HID = 2
        EMBED_HID_SZ = 2 * dim.sysid

        # policy
        with tf.variable_scope("pol"):
            if flavor == BLIND:
                policy_input = obz
                self.sysid_err_supervised = tf.constant(0.0)
                self.embed = tf.constant(0.0)
            elif flavor == PLAIN:
                self.sysid_err_supervised = tf.losses.mean_squared_error(
                    tf.stop_gradient(sysidz), self.traj2sysid)
                policy_input = tf.concat([obz, self.traj2sysid]) if test else obz_all
            elif flavor == EXTRA:
                sysid_processor_input = self.traj2sysid if test else sysidz
                sysid_processor = MLPModule(sysid_processor_input,
                    EMBED_N_HID, EMBED_HID_SZ, dim.embed, "sysid_processor")
                self.embed = sysidz
                policy_input = tf.concat([obz, sysid_processor], axis=1, name="input_concat")
                self.sysid_err_supervised = tf.losses.mean_squared_error(
                    tf.stop_gradient(sysidz), self.traj2sysid)
            elif flavor == EMBED:
                embed_mean = MLPModule(sysidz, EMBED_N_HID, EMBED_HID_SZ, dim.embed, "embed")
                # TODO does this need to be stochastic?
                self.embed_logstd = tf.constant(-2.0)
                #self.embed_logstd = tf.maximum(
                    #tf.get_variable(name="embed_logstd", shape=[1, dim.embed],
                    #initializer=tf.constant_initializer(-0.3)),
                    #-2.0)
                self.embed_pd = tf.distributions.Normal(loc=embed_mean, scale=embed_logstd)
                self.embed = self.embed_pd.sample()

                embed_input = self.traj2embed if test else self.embed
                policy_input = tf.concat([obz, embed_input], axis=1, name="input_concat")
                self.sysid_err_supervised = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.embed), self.traj2embed)

                #self.extra_rewards.append(0.005 * tf.reduce_mean(self.embed_logstd))
                #self.extra_reward_names.append("embed entropy")

                #mean, var = tf.nn.moments(self.embed, 0)
                #dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))
                #std_dist = tf.distributions.Normal(loc=0.0, scale=1.0)
                #embed_KL = tf.reduce_mean(tf.distributions.kl_divergence(dist, std_dist))
                #self.extra_rewards.append(-0.1 * embed_KL)
                #self.extra_reward_names.append("neg_embed_KL")
            elif flavor == TRAJ:
                self.traj_conv = sysid_convnet(trajs, dim.embed)
                policy_input = tf.concat([obz, self.traj_conv], axis=1, name="input_concat")
                self.sysid_err_supervised = tf.constant(0.0)
            else:
                raise ValueError("flavor '{}' does not exist".format(flavor))

            self.Q_rms = RunningMeanStd(shape=(1,))
            policy_input = tf.concat([policy_input, self.in_ac], axis=1, name="Q_input")
            print("Q input dimensionality:", policy_input.get_shape().as_list())
            self.Q = tf.squeeze(MLPModule(policy_input, n_hid, hid_size, 1, "Q"))



    # given the actual dynamics parameters, compute the embedding
    def sysid_to_embedded(self, sysid_vals):
        if self.flavor in [BLIND, TRAJ]:
            # could also just return sysid_vals, but this draws attention to lack of sysid
            #return 0 * sysid_vals
            return np.zeros((sysid_vals.shape[0], 1))

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

    def set_is_train(self, is_train):
        self.is_train.assign(is_train)

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


    def act_traj(self, stochastic, ob, ob_traj, ac_traj):
        raise NotImplementedError

    # act - ob is concat(ob, sysid)
    @profile
    def act(self, _, ob):

        # assuming actions are roughly unit normal! and over-cover a bit
        # TODO: warm start?
        N = ob.shape[0]
        C = self.cem_count
        K = self.cem_keep

        ob_expanded = np.tile(ob[:,None,:], (1, C, 1)).reshape((-1, self.dim.ob_concat))
        #ac_sets = np.random.normal(size=(N, C, self.dim.ac))
        ac_sets = self.uniform_sampler.sample(self.npr, (N, C, self.dim.ac))
        sess = tf.get_default_session()

        mu = None
        std = None

        iter = 1
        while iter <= self.cem_iters:
            # find the best points in this batch.
            feed = {
                self.in_ob : ob_expanded,
                self.in_ac : ac_sets.reshape((-1, self.dim.ac)),
            }
            Qvals = sess.run(self.Q, feed_dict=feed).reshape((N, C))
            part = np.argpartition(Qvals, C-K, axis=1)
            ibest = part[:,-K:]
            # transposition determined by trial and error. lol
            best = ac_sets[np.arange(N), ibest.T, :].transpose([1, 0, 2])

            # sample new batch.
            if mu is None:
                mu = np.mean(best, axis=1)
                std = np.std(best, axis=1)
            else:
                mu = self.cem_smooth * mu + (1.0 - self.cem_smooth) * np.mean(best, axis=1)
                std = self.cem_smooth * std + (1.0 - self.cem_smooth) * np.std(best, axis=1)

            ac_sets = (mu[:,None,:] + std[:,None,:] *
                # np.random.normal(size=(N, C, self.dim.ac))
                self.normal_sampler.sample(self.npr, (N, C, self.dim.ac))
            )
            iter += 1

        qvals = sess.run(self.Q, feed_dict={self.in_ob : ob, self.in_ac : mu})
        embed = self.sysid_to_embedded(ob[:,self.dim.ob:])

        # plot q func. only works if action is 1D
        if True and np.random.uniform() < 5e-3:
            ac_sweep = np.linspace(-1, 1, 1000)[:,None]
            ob_rep = np.tile(ob[0,:], (1000, 1))
            feed = {
                self.in_ob : ob_rep,
                self.in_ac : ac_sweep,
            }
            Qvals = sess.run(self.Q, feed_dict=feed)
            chosen = mu[0]
            val = qvals[0]
            plt.clf()
            plt.plot(ac_sweep, Qvals)
            plt.scatter(chosen, val, color=(0,0,0))
            #plt.show(block=False)
            plt.show(block=False)
            x=ob[0]
            plt.title(f"x={x[0]:.2f}, dx={x[1]:.2f}, th={x[2]:.2f}, dth={x[3]:.2f}")
            plt.pause(0.001)

        return mu, qvals, embed


    def ob_mean_std(self):
        sess = tf.get_default_session()
        vars = (self.ob_rms.mean, self.ob_rms.std)
        return sess.run(vars)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
