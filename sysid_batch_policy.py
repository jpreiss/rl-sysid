import tensorflow as tf
import numpy as np

from sysid_utils import Dim, MLP, SquashedGaussianPolicy, minibatch_iter


# policy flavors:

# no access to SysID values - domain randomization only
BLIND = "blind"

# policy has no access to SysID, but value fns do - for debugging only
BLIND_POLICY_PLAIN_VF = "blind_policy_plain_vf"

# MLP, access to true SysID values concatenated w/ obs
PLAIN = "plain"

# basically same as Plain, but with same net structure as Embed.
# still try to estimate true SysID from trajectories, but it's "preprocessed".
# exists to experimentally verify that any diff btw. Plain-style and Embed
# is due to architecture and having more layers / parameters.
EXTRA = "extra"

# "ours". converts SysID to embedding.
# estimator tries to identify this embedding instead of true SysID.
EMBED = "embed"


class SysIDPolicy(object):

    flavors = [BLIND, BLIND_POLICY_PLAIN_VF, PLAIN, EXTRA, EMBED]

    # set up the network
    # NOTE: due to normalization of SysID values and KL-regularization of embedding space,
    # alpha_sysid shouldn't need to vary between environments - but we'll see...
    #
    # expects you to construct a variable scope for reusing, etc.
    def __init__(self, ob_input, ob_traj_input, ac_traj_input, dim,
                 flavor, hid_sizes, embed_hid_sizes, activation,
                 logstd_is_fn,
                 alpha_sysid, embed_KL_weight, seed,
                 test):

        if flavor not in SysIDPolicy.flavors:
            raise ValueError(f"flavor '{flavor}' does not exist")

        if test:
            #print("constructing policy using test input.")
            pass

        # we store the flavor here for logging purposes, but RL algos should never read it
        self.flavor = flavor
        self.dim = dim
        self.alpha_sysid = alpha_sysid
        self.ob = ob_input

        # placeholders
        ob, sysid = tf.split(ob_input, [dim.ob, dim.sysid], axis=-1)
        #self.ob_traj = tf.placeholder(tf.float32, (None, dim.window, dim.ob), name="ob_traj")
        #self.ac_traj = tf.placeholder(tf.float32, (None, dim.window, dim.ac), name="ac_traj")
        self.ob_traj = ob_traj_input
        self.ac_traj = ac_traj_input
        trajs = tf.concat([self.ob_traj, self.ac_traj], axis=-1)

        self.extra_rewards = []
        self.extra_reward_names = []
        self.logs = []

        EMPTY_TENSOR = tf.constant(np.zeros((1,0)), name="EMPTY_TENSOR")

        # set up estimator.
        with tf.variable_scope("estimator"):
            if flavor in [BLIND, BLIND_POLICY_PLAIN_VF]:
                self.estimator = EMPTY_TENSOR
            elif flavor in [PLAIN, EXTRA]:
                self.estimator = sysid_convnet(trajs, dim.sysid)
            elif flavor == EMBED:
                self.estimator = sysid_convnet(trajs, dim.embed)
            estimator_scope = tf.get_variable_scope().name

        # setup policy. for {EXTRA, EMBED}, embedder is a part of the policy
        with tf.variable_scope("policy"):
            if flavor == BLIND:
                self.est_target = EMPTY_TENSOR
                pol_input = None
                vf_input = None

            elif flavor == BLIND_POLICY_PLAIN_VF:
                self.est_target = EMPTY_TENSOR
                pol_input = None
                vf_input = sysid

            elif flavor == PLAIN:
                self.est_target = sysid
                pol_input = self.estimator if test else sysid
                vf_input = sysid

            elif flavor == EXTRA:
                self.est_target = sysid
                sysid_val = self.estimator if test else sysid
                embedder = MLP(sysid_val, "sysid_processor",
                    embed_hid_sizes, dim.embed, activation=activation)
                pol_input = tf.nn.relu(embedder.out)
                vf_input = sysid

            elif flavor == EMBED:
                self.est_target = embedder = MLP("embedder", sysid,
                    embed_hid_sizes, dim.embed, activation=activation).out
                embed_KL = tf.reduce_mean(kl_from_unit_normal(embedder))
                self.extra_rewards.append(-embed_KL_weight * embed_KL)
                self.extra_reward_names.append("neg_embed_KL")
                pol_input = tf.nn.relu(self.estimator if test else embedder)
                vf_input = tf.nn.relu(embedder)
                tf.summary.histogram("embeddings", embedder)
                tf.summary.scalar("embed_KL", embed_KL)

            pol_input = concat_notnone([ob, pol_input], axis=-1, name="pol_input")
            policy = SquashedGaussianPolicy("policy", pol_input,
                hid_sizes, dim.ac, tf.nn.relu,
                logstd_is_fn=logstd_is_fn,
                seed=seed)
            self.logs.append((tf.reduce_mean(policy.std), "mean_action_stddev"))
            self.ac_stochastic = policy.ac
            self.ac_mean = policy.mu
            self.reg_loss = policy.reg_loss
            self.entropy = policy.entropy
            self.pdf = policy

            # compute the log-probability of stochastic actions under policy.
            # note: does not stop gradient!!
            with tf.variable_scope("log_prob"):
                self.log_prob = policy.logp_raw(policy.raw_ac)

            pol_scope = tf.get_variable_scope().name


        # estimator L2 training loss.
        with tf.variable_scope("estimator_loss"):
            self.estimator_loss = tf.losses.mean_squared_error(
                self.estimator, tf.stop_gradient(self.est_target))
            self.logs.append((tf.sqrt(self.estimator_loss), "rmse_estimator"))

        # do not compute the value function here, but specify what its input should be.
        # lets the RL algorithm to use different kinds of value functions e.g. Q vs. V
        self.vf_input = concat_notnone([ob, vf_input], axis=-1)

        # store variable collections for tf optimizers.
        self.policy_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, pol_scope)
        self.estimator_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, estimator_scope)
        self.all_vars = self.policy_vars + self.estimator_vars


    def logp(self, actions):
        return self.pdf.logp(0.9999 * actions)


    # given the ob/ac trajectories, estimates:
    #
    # PLAIN, EXTRA : the SysID values
    # EMBED        : the embedding
    # BLIND        : empty tensor
    #
    # does its own minibatching to handle huge inputs. (TODO: externalize)
    def estimate_sysid(self, sess, ob_trajs, ac_trajs):

        if ob_trajs.shape[0] <= 2048:
            return sess.run(self.estimator, {
                self.ob_traj : ob_trajs,
                self.ac_traj : ac_trajs,
            })
        else:
            # split and recurse
            return np.vstack(list(
                self.estimate_sysid(sess, o, a)
                for o, a in minibatch_iter(2048, ob_trajs, ac_trajs)
            ))


# construct the 1D convolutional network for (ob, ac)^k -> SysID
def sysid_convnet(input, sysid_dim):
    conv1 = tf.layers.conv1d(input, filters=64, kernel_size=3, activation=None)
    conv2 = tf.layers.conv1d(conv1, filters=64, kernel_size=3, activation=tf.nn.relu)
    # first 2 conv layers without nonlinearity is equivalent to one bigger conv layer
    # but has better learning properties (TODO: find citation)
    conv3 = tf.layers.conv1d(conv2, filters=64, kernel_size=3, activation=tf.nn.relu)
    flat = tf.layers.flatten(conv3)
    #print("convnet flat dim:", flat.shape[1])
    flat2 = tf.layers.dense(flat, 128, activation=tf.nn.relu)
    out = tf.layers.dense(flat2, sysid_dim)
    return out


def kl_from_unit_normal(x):
    mean, var = tf.nn.moments(x, axes=0)
    dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))
    std_dist = tf.distributions.Normal(loc=0.0, scale=1.0)
    kl = tf.reduce_mean(tf.distributions.kl_divergence(dist, std_dist))
    return kl

def concat_notnone(tensors, *args, **kwargs):
    return tf.concat([t for t in tensors if t is not None], *args, **kwargs)
