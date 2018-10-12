from collections import namedtuple
import itertools as it
import os

import numpy as np
import tensorflow as tf


# ob: observation without sysid added
# sysid: mass, inertia, etc...
# ob_concat: ob + sysid
# ac: action
# embed: dimensionality of embedding space
# agents: number of agents in batch environment
# window: length of rolling window for sysid network input
Dim = namedtuple('Dim', 'ob sysid ob_concat ac embed agents window')

RENDER_EVERY = 10

# for fixed length episodes
# expects env to have ep_len member variable
def sysid_simple_generator(sess, pi, env, stochastic, test=False, force_render=None, callback=None):

    N = env.N
    dim = pi.dim
    horizon = env.ep_len

    #pi.set_is_train(not test)

    # Initialize history arrays
    obs = np.zeros((horizon, N, dim.ob_concat))
    acs = np.zeros((horizon, N, dim.ac))
    acmeans = np.zeros((horizon, N, dim.ac))
    aclogstds = np.zeros((horizon, N, dim.ac))
    logps = np.zeros((horizon, N))
    embeds = np.zeros((horizon, N, pi.est_target.shape[-1]))
    rews = np.zeros((horizon, N))
    # rolling window, starting with zeros
    ob_trajs = np.zeros((horizon, N, dim.window, dim.ob))
    ac_trajs = np.zeros((horizon, N, dim.window, dim.ac))

    npr = env.np_random

    uninit = tf.report_uninitialized_variables().shape.num_elements()
    if uninit and uninit > 0:
        print("sysid_simple_generator is initializing global variables")
        sess.run(tf.global_variables_initializer())

    for episode in it.count():

        # TODO could make it possible to include more than one reset in a batch
        # without also resampling SysIDs. But is it actually useful?
        env.sample_sysid()
        ob = env.reset()
        assert ob.shape == (N, dim.ob_concat)
        ob_trajs *= 0
        ac_trajs *= 0

        # touch / rm this file to toggle rendering
        render = (force_render if force_render is not None
            else os.path.exists("render"))

        for step in range(horizon):

            if render and episode % RENDER_EVERY == 0:
                env.render()

            obs[step,:,:] = ob

            # TODO consider if stochastic or not?
            target = pi.ac_stochastic if stochastic else pi.ac_mean
            feed = { pi.ob : ob }
            if test:
                feed = {**feed, **{
                    pi.ob_traj : ob_trajs[step],
                    pi.ac_traj : ac_trajs[step],
                }}
            # dat tight coupling
            ac, embed, logp, acmean, aclogstd = sess.run([
                target, pi.est_target, pi.log_prob, pi.ac_mean, pi.ac_logstd], feed)

            # epsilon-greedy exploration (TODO: pass in params)
            #rand_acts = np.random.uniform(-1.0, 1.0, size=ac.shape)
            #epsilon = np.random.uniform(size=ac.shape[0])
            #greedy = epsilon < 0.4
            #ac[greedy] = rand_acts[greedy]

            acs[step,:,:] = ac
            acmeans[step,:,:] = acmean
            aclogstds[step,:,:] = aclogstd
            assert logp.size == N
            logps[step,:] = logp
            embeds[step,:,:] = embed

            if step < horizon - 1:
                ob_trajs[step+1] = np.roll(ob_trajs[step], -1, axis=1)
                ac_trajs[step+1] = np.roll(ac_trajs[step], -1, axis=1)
                ob_trajs[step+1,:,-1,:] = ob[:,:dim.ob]
                ac_trajs[step+1,:,-1,:] = ac


            ob_next, rew, _, _ = env.step(ac)
            rews[step,:] = rew

            if callback:
                callback(locals(), globals())

            ob = ob_next

        # Episode over.

        # in console we want to print the task reward only
        ep_rews = np.sum(rews, axis=0)

        # evaluate SysID errors and add to the main rewards.
        sysids = obs[0,:,dim.ob:]
        assert np.all((sysids[None,:,:] == obs[:,:,dim.ob:]).flat)
        embed_trues = sess.run(pi.est_target, { pi.ob : obs[0,:,:] })

        try:
            embed_estimates = pi.estimate_sysid(sess,
                ob_trajs.reshape((horizon * N, dim.window, dim.ob)),
                ac_trajs.reshape((horizon * N, dim.window, dim.ac)))
            embed_estimates = embed_estimates.reshape((horizon, N, -1))
            err2s = (embeds - embed_estimates) ** 2
            assert len(err2s.shape) == 3
            if err2s.shape[-1] == 0:
                meanerr2s = np.zeros(err2s.shape[:-1])
            else:
                meanerr2s = np.mean(err2s, axis=-1)
            # apply the err2 for each window to *all* actions in that window
            sysid_loss = 0 * rews
            for i in range(horizon):
                begin = max(i - dim.window, 0)
                sysid_loss[begin:i,:] += meanerr2s[i,:]
            sysid_loss *= (pi.alpha_sysid / dim.window)
            total_rews = rews - sysid_loss
            # TODO keep these separate and let the RL algorithm reason about it?

        except NotImplementedError:
            # e.g. exploration policy - just make up garbage
            embed_estimates = embed_trues + np.nan
            meanerr2s = None
            sysid_loss = np.nan
            total_rews = rews

        # yield the batch to the RL algorithm
        yield {
            "ob" : obs, "ac" : acs, "logp" : logps, "acmean": acmeans, "aclogstd": aclogstds,
            "rew" : total_rews, "task_rews" : rews,
            "ob_traj" : ob_trajs, "ac_traj" : ac_trajs,
            "ep_rews" : ep_rews, "ep_lens" : horizon + 0 * ep_rews,
            "est_true" : embeds, "est" : embed_estimates,
            "est_mserr" : meanerr2s, "sysid_loss" : sysid_loss,
        }


def add_vtarg_and_adv(seg, gamma, lam):
    rew = seg["rew"]
    vpred = seg["vpred"]
    T, N = rew.shape
    # making the assumption that vpred is a smooth function of (non-sysid) state
    # and the error here is small
    # also assuming no special terminal rewards
    vpred = np.vstack((vpred, vpred[-1,:]))
    gaelam = np.zeros((T + 1, N))
    for t in reversed(range(T)):
        delta = rew[t] + gamma * vpred[t+1] - vpred[t]
        gaelam[t] = delta + gamma * lam * gaelam[t+1]
    vpred = vpred[:-1]
    gaelam = gaelam[:-1]
    seg["adv"] = gaelam
    seg["tdlamret"] = gaelam + vpred


# flattens arrays that are (horizon, N, ...) shape into (horizon * N, ...)
def seg_flatten_batches(seg, keys=None):
    if keys is None:
        keys = ("ob", "ac", "logp", "task_rews",
                "ob_traj", "ac_traj", "est_true",
                "adv", "tdlamret", "vpred")
    for s in keys:
        sh = seg[s].shape
        if len(sh) > 1:
            newshape = [sh[0] * sh[1]] + list(sh[2:])
            seg[s] = np.reshape(seg[s], newshape)


class ReplayBuffer(object):
    def __init__(self, N, dims):
        N = int(N)
        def arrdim(d):
            if type(d) is tuple:
                return (N,) + d
            else:
                return (N, d)
        self.bufs = tuple(np.zeros(arrdim(d)) for d in dims)
        self.N = N
        self.size = 0
        self.cursor = 0


    def add(self, *args):
        if self.size < self.N:
            self.size += 1
        if self.cursor == 0:
            #print("replay buffer roll over")
            pass
        for buf, item in zip(self.bufs, args):
            buf[self.cursor] = item
        self.cursor = (self.cursor + 1) % self.N


    def add_batch(self, *args):

        K = args[0].shape[0]
        assert K < self.N
        new_end = self.cursor + K

        if new_end <= self.N:
            # normal case, batch fits in buffer
            for buf, batch in zip(self.bufs, args):
                buf[self.cursor:new_end] = batch
            self.cursor = new_end % self.N
            self.size = max(self.size, new_end)
        else:
            # special case, batch wraps around buffer.
            # split and recurse.
            split = self.N - self.cursor
            self.add_batch(*(arg[:split] for arg in args))
            assert self.cursor == 0
            assert self.size == self.N
            self.add_batch(*(arg[split:] for arg in args))


    def sample(self, np_random, batch_size):
        idx = np_random.randint(self.size, size=batch_size)
        returns = [buf[idx] for buf in self.bufs]
        return returns


class MLP(object):
    def __init__(self, name, input, hid_sizes, output_size, activation, reg=None, reuse=False):
        x = input
        with tf.variable_scope(name):
            for i, size in enumerate(hid_sizes):
                x = tf.layers.dense(x, size, activation=activation,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=reg, bias_regularizer=reg,
                    reuse=reuse, name="fc_{}".format(i))
            self.out = tf.layers.dense(x, output_size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=reg,
                #use_bias=False,
                name="fc_out", reuse=reuse)
            if output_size == 1:
                self.out = self.out[:,0]
            # TODO: seems circular, can we get this without using strings?
            scope_name = tf.get_variable_scope().name
            self.scope = scope_name
            self.vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)


class SquashedGaussianPolicy(object):
    def __init__(self, name, input, hid_sizes, output_size, activation,
                 logstd_is_fn, seed, reg=None, reuse=False):

        self.name = name

        if logstd_is_fn:
            self.mlp = MLP(name, input, hid_sizes, 2*output_size, activation, reg, reuse)
            self.mlp.out *= 0.1
            self.mu, logstd = tf.split(self.mlp.out, 2, axis=1)
            self.logstd = tf.clip_by_value(logstd, -2.0, 2.0)
        else:
            self.mlp = MLP(name, input, hid_sizes, output_size, activation, reg, reuse)
            self.mlp.out *= 0.1
            self.mu = self.mlp.out
            init_logstd = -0.3 * np.ones(output_size, dtype=np.float32)
            self.logstd = tf.get_variable("logstd", initializer=init_logstd)

        self.std = tf.exp(self.logstd)
        self.pdf = tf.distributions.Normal(loc=self.mu, scale=self.std)
        self.raw_ac = self.pdf.sample(seed=seed)
        self.ac = tf.tanh(self.raw_ac)

        with tf.variable_scope("squashed_entropy_bound"):
            squash_correction = tf.log(1.0 - tf.tanh(self.mu) ** 2)
        self.entropy = self.pdf.entropy()
        self.entropy = tf.reduce_sum(self.entropy + squash_correction, axis=-1)

        self.reg_loss = 5e-4 * (
            tf.reduce_mean(self.logstd ** 2) + tf.reduce_mean(self.mu ** 2))


    def logp(self, actions):
        return self.logp_raw(tf.atanh(actions))

    # actions should be raw_ac, with tanh not applied
    def logp_raw(self, raw_actions):
        log_p = -(0.5 * tf.to_float(raw_actions.shape[-1]) * np.log(2 * np.pi)
            + tf.reduce_sum(self.logstd, axis=-1)
            + 0.5 * tf.reduce_sum(((raw_actions - self.mu) / self.std) ** 2, axis=-1)
        )
        EPS = 1e-6
        squash_correction = tf.reduce_sum(tf.log(1.0 - tf.tanh(raw_actions)**2 + EPS), axis=1)
        return log_p - squash_correction

    def get_params_internal(self):
        return self.mlp.vars


def minibatch_iter(size, *args, np_random=None):
    N = args[0].shape[0]
    if np_random is not None:
        order = np_random.permutation(N)
    else:
        order = np.arange(N)
    for k in range(0, N - size + 1, size):
        yield (a[order[k:(k+size)]] for a in args)
    if N % size != 0:
        start = size * (N // size)
        yield (a[order[start:]] for a in args)


# TODO is there a TF op for this?
def lerp(a, b, theta):
    return (1.0 - theta) * a + theta * b


def printstats(var, name):
    print("{}: mean={:3f}, std={:3f}, min={:3f}, max={:3f}".format(
        name, np.mean(var), np.std(var), np.min(var), np.max(var)))


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def fmt_row(width, row, header=False):

    def fmt_item(x, l):
        if isinstance(x, np.ndarray):
            assert x.ndim==0
            x = x.item()
        if isinstance(x, float): rep = "%g"%x
        else: rep = str(x)
        return " "*(l - len(rep)) + rep

    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-"*len(out)
    return out

