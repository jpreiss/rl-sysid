#!/usr/bin/env python3

from copy import deepcopy
from itertools import *
import os
import sys
import collections
import multiprocessing
import pdb
import pickle

import gym
import logging

from baselines.common import set_global_seeds
from baselines.trpo_mpi import trpo_batch
from baselines.ppo1 import pposgd_batch
import baselines.common.tf_util as U
import baselines.common.batch_util2 as batch2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import scipy as sp


import sysid_batch_policy
from sysid_batch_policy import SysIDPolicy, Dim


# JSON schema for fully defining experiments
spec_prototype = {
    "env" : "HalfCheetah-Batch-v1",
    "n_batch" : 64,
    "n_total" : 8,
    "randomness" : 1.75,

    "test_mode": "tweak", # one of "resample" or "tweak"
    "test_tweak" : 1.1,

    "flavors" : ["blind", "extra", "embed"],
    "alphas" : [0.0, 0.01, 0.03, 0.1],
    "seeds" : [0],

    "algorithm" : "ppo",
    "opt_iters" : 2,
    "opt_batch" : 256,
    "learning_rate" : 1e-3,
    "lr_schedule" : "linear",
    "entropy_coeff" : 0.010,
    "train_iters" : 200,
    #"tdlambda" : 0.96,

    "embed_dim" : 8,
    "window" : 16,

    "n_hidden" : 3,
    "hidden_sz" : 128,
    "activation" : "relu",
}

def make(spec, **kwargs):

    kwargs = deepcopy(kwargs)
    for key in ["n_batch", "n_total", "randomness"]:
        kwargs[key] = spec[key]

    spec = gym.envs.registry.env_specs[spec["env"]]
    return spec.make(**kwargs)


# convert the experiment spec into a string that's a valid directory name
# TODO: this is hacky
def spec_slug(spec):
    def kvstr(k, v):
        k_short = "_".join(w[:3] for w in k.split("_"))
        v_sanitized = str(v).replace(".", "p").replace("-", "_") \
            .replace(" ", "_").replace(",", "").replace("'", "") \
            .replace("[", "").replace("]", "")
        return k_short + "_" + v_sanitized
    slug = "_".join([kvstr(k, v) for k, v in sorted(spec.items())])
    return slug

class SegArray(object):
    def __init__(self, segs):
        self.segs = segs

    def find(self, flavor, alpha, seed=None):
        for f, a, seed_segs in self.segs:
            if f == flavor and a == alpha:
                if seed is None:
                    return seed_segs
                else:
                    for s, segs in seed_segs:
                        if s == seed:
                            return seed_segs


# directory for the specific flavor/alpha combination
def dir_fn(spec, flavor, alpha):
    return os.path.join("./results", spec_slug(spec), flavor, "alpha" + str(alpha))

# directory for the flavor/alpha/seed combination
def seed_csv_dir(spec, flavor, alpha, seed):
    basedir = dir_fn(spec, flavor, alpha)
    return os.path.join(basedir, str(seed), "train_log")

# TODO make "progress.csv" not hard coded in the training function
def seed_csv_path(spec, flavor, alpha, seed):
    return os.path.join(seed_csv_dir(spec, flavor, alpha, seed), "progress.csv")


# for compatibility with OpenAI Baselines learning algorithms
def make_batch_policy_fn(np_random, spec, env, flavor, alpha_sysid):
    def f(name, ob_space, ac_space):
        sysid_dim = int(env.sysid_dim)
        for space in (ob_space, ac_space):
            assert isinstance(space, gym.spaces.Box)
            assert len(space.shape) == 1
        dim = Dim(
            ob = ob_space.shape[0] - sysid_dim,
            sysid = sysid_dim,
            ob_concat = ob_space.shape[0],
            ac = ac_space.shape[0],
            embed = spec["embed_dim"],
            agents = env.N,
            window = spec["window"],
        )
        return SysIDPolicy(np_random=np_random, name=name, flavor=flavor, dim=dim,
            hid_size=spec["hidden_sz"], n_hid=spec["n_hidden"],
            alpha_sysid=alpha_sysid)
    return f


def train(spec, sess, flavor, alpha_sysid, seed, csvdir):

    env_id = spec["env"]
    env = make(spec)
    env.seed(seed)

    var_init_npr = np.random.RandomState(seed)
    set_global_seeds(seed)

    policy_fn = make_batch_policy_fn(var_init_npr, spec, env, flavor, alpha_sysid)

    gym.logger.setLevel(logging.WARN)

    algo = spec["algorithm"]
    if algo == "trpo":
        #raise NotImplementedError
        trained_policy = trpo_batch.learn(env, policy_fn,
            max_iters=spec["train_iters"],
            max_kl=0.003, cg_iters=10, cg_damping=0.1,
            gamma=0.99, lam=0.98,
            vf_iters=spec["opt_iters"], vf_stepsize=spec["learning_rate"],
            entcoeff=spec["entropy_coeff"],
            logdir=csvdir)
    elif algo == "ppo":
        trained_policy = pposgd_batch.learn(env.np_random, env, policy_fn,
            max_iters=spec["train_iters"],
            clip_param=0.2, entcoeff=spec["entropy_coeff"],
            optim_epochs=spec["opt_iters"], optim_batchsize=spec["opt_batch"],
            optim_stepsize=spec["learning_rate"],
            gamma=0.99, schedule=spec["lr_schedule"],
            #lam=spec["tdlambda"],
            lam=0.96,
            logdir=csvdir
        )
    else:
        assert False, "invalid choice of RL algorithm: " + algo
    env.close()
    rews = env.mean_rews
    return rews


# load the policy and test, return array of "seg" dictionaries
def test(spec, sess, env, flavor, seed, mydir, n_sysid_samples):

    delta_seed = 100 if spec["test_mode"] == "resample" else 0
    env.seed(seed + delta_seed)

    var_init_npr = np.random.RandomState(seed + delta_seed)
    set_global_seeds(seed + delta_seed)

    alpha_sysid = 0 # doesn't matter at test time
    pi = make_batch_policy_fn(var_init_npr, spec, env, flavor, alpha_sysid)(
        "pi", env.observation_space, env.action_space)

    seeddir = os.path.join(mydir, str(seed))
    ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    # while in some cases, you might set stochastic=False at test time
    # to get the "best" actions, for SysID stochasticity could be
    # an important / desired part of the policy
    seg_gen = batch2.sysid_simple_generator(pi, env, stochastic=True, test=True)
    segs = list(islice(seg_gen, n_sysid_samples))
    return segs


# train the policy, saving the training logs and trained policy
# TODO this function can probably be eliminated
def train_one_flavor(spec, flavor, alpha_sysid, mydir):
    os.makedirs(mydir, exist_ok=True)
    n_seeds = len(spec["seeds"])
    for i, seed in enumerate(spec["seeds"]):
        print("training {}, alpha = {}, seed = {} ({} out of {} seeds)".format(
            flavor, alpha_sysid, seed, i + 1, n_seeds))
        status = "{} | alpha = {} | seed {}/{}".format(
            flavor, alpha_sysid, i + 1, n_seeds)
        set_xterm_title(status)
        seeddir = os.path.join(mydir, str(seed))
        csvdir = os.path.join(seeddir, 'train_log')
        os.makedirs(csvdir, exist_ok=True)

        g = tf.Graph()
        U.flush_placeholders() # TODO get rid of U
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=g, config=config) as sess:
            mean_rews = train(spec, sess, flavor, alpha_sysid, seed, csvdir)
            ckpt_path = os.path.join(seeddir, "trained_model.ckpt")
            rews_path = os.path.join(seeddir, "mean_rews.pickle")
            with open(rews_path, "wb") as f:
                pickle.dump(mean_rews, f, protocol=4)
            saver = tf.train.Saver()
            saver.save(sess, ckpt_path)


def render_different_envs(spec, flavor, alpha, seed, mydir):

    env = make(spec)
    g = tf.Graph()
    U.flush_placeholders()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g, config=config) as sess:
        segs = test(spec, sess, env, flavor, seed, mydir)

    rews = np.array([list(chain(iter_key(segs, "ep_rews")))])
    n_eps, N = rews.shape
    assert N == env.N
    mean_rew = np.mean(rews, axis=0)
    sort_ind = np.argsort(mean_rew)

    for i in range(0, N, N // 5):
        pass
    # TODO not implemented yet!!!!
    # This function is supposed to help visualize the good vs. bad random envs


def test_one_flavor(spec, flavor, alpha, mydir, sysid_iters):

    tweak = spec["test_tweak"] if spec["test_mode"] == "tweak" else 0.0
    env = make(spec, tweak=tweak)

    def test_one(flavor, alpha, seed):
        status = "{} | alpha = {} | seed {}/{}".format(
            flavor, alpha, seed + 1, len(spec["seeds"]))
        set_xterm_title(status)
        g = tf.Graph()
        U.flush_placeholders()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=g, config=config) as sess:
            return test(spec, sess, env, flavor, seed, mydir, sysid_iters)

    return [(seed, test_one(flavor, alpha, seed)) for seed in spec["seeds"]]


# helper fn for parallelization over flavors * alphas.
# arg_fn should take (flavor, alpha) and return args for fn(*args).
# returns list of (flavor, alpha, fn result).
def grid(spec, fn, arg_fn, n_procs):
    params = list(product(spec["flavors"], spec["alphas"]))
    if n_procs == 1 or len(params) == 1:
        return [(flavor, alpha, fn(*arg_fn(flavor, alpha))) for flavor, alpha in params]
    else:
        asyncs = []
        pool = multiprocessing.Pool(processes=n_procs)
        for flavor, alpha in params:
            args = arg_fn(flavor, alpha)
            res = pool.apply_async(fn, args)
            asyncs.append(res)
        results = [(flavor, alpha, res.get()) for (flavor, alpha), res in zip(params, asyncs)]
        return results


def train_all(spec, n_procs):
    def train_arg_fn(flavor, alpha):
        mydir = dir_fn(spec, flavor, alpha)
        return spec, flavor, alpha, mydir
    grid(spec, train_one_flavor, train_arg_fn, n_procs=n_procs)


def test_all(spec, sysid_iters, n_procs):
    def test_arg_fn(flavor, alpha):
        mydir = dir_fn(spec, flavor, alpha)
        return spec, flavor, alpha, mydir, sysid_iters
    segs = grid(spec, test_one_flavor, test_arg_fn, n_procs=n_procs)
    return segs


def iter_key(dicts, key):
    for d in dicts:
        yield d[key]

def types_str(x):
    inner = lambda: ", ".join(types_str(y) for y in x)
    if type(x) == type(()):
        return "(" + inner() + ")"
    if type(x) == type([]):
        return "[" + inner() + "]"
    return str(type(x))

# output shape: (flavor*alpha, seed, iters, N)
def flat_rewards(segs):
    def all_flat_rews(seed_segs):
        return [list(chain(iter_key(segs, "ep_rews"))) for seed, segs in seed_segs]
    all_rews = np.array([all_flat_rews(seed_segs) for _, _, seed_segs in segs])
    return all_rews

def sysid_err(segs):
    trues = np.vstack(x[None,...] for x in iter_key(segs, "embed_true"))
    estimates = np.vstack(x[None,...] for x in iter_key(segs, "embed_estimate"))
    return np.mean((trues[:,None,...] - estimates) ** 2)

def print_test_results(segs, flat_rews):
    for (flavor, alpha, seed_segs), rews in zip(segs, flat_rews):
        #print("print_test flat_rews for {}, {}: {}".format(flavor, alpha, rews))
        def gen_lines():
            yield "{}, alpha_sysid = {}:".format(flavor, alpha)
            for (seed, seed_seg), r in zip(seed_segs, rews):
                yield " seed {}: mean: {:.2f}, std: {:.2f}, sysid err2: {:.3f}".format(
                    seed, np.mean(r), np.std(r), sysid_err(seed_seg))
            yield "overall: mean: {:.2f}, std: {:.2f}".format(
                    np.mean(rews), np.std(rews))
        boxprint(list(gen_lines()))


def print_anova(flat_rews):
    # for ANOVA
    each_trial_rews = np.reshape(flat_rews, (flat_rews.shape[0], -1))
    f, p = sp.stats.f_oneway(*each_trial_rews)
    print("ANOVA each trial results: f = {}, p = {}".format(f, p))

    each_seed_rews = np.mean(flat_rews, axis=2).reshape((flat_rews.shape[0], -1))
    f, p = sp.stats.f_oneway(*each_seed_rews)
    print("ANOVA individual seed results: f = {}, p = {}".format(f, p))


def load_learning_curve(spec, flavor, alpha, seed):
    path = seed_csv_path(spec, flavor, alpha, seed)
    data = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64)
    # TODO don't hard-code 64
    col_names = ["Env{}Rew".format(i) for i in range(32)]
    rews = np.column_stack([data[c] for c in col_names])
    return rews
    #return data["EpRewMean"]

# return has dimensionality (seed, timestep, N)
def load_all_learning_curves(spec, flavor, alpha):
    return np.stack([load_learning_curve(spec, flavor, alpha, seed=s) for s in spec["seeds"]])

def load_env_mean_rewards(spec, flavor, alpha, seed):
    dir = dir_fn(spec, flavor, alpha)
    path = os.path.join(dir, str(seed), "mean_rews.pickle")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")

def embed_scatter_data(segs):
    trues = np.stack(iter_key(segs, "embed_true"))
    estimates = np.stack(iter_key(segs, "embed_estimate"))
    return trues, estimates


# compute the policy mean action at a fixed task state
# conditioned on the SysID params
def action_conditional(sess, env_id, test_state, flavor, seed, mydir):

    assert env_id == "PointMass-Batch-v0"
    env = gym.make(env_id)

    alpha_sysid = 0 # doesn't matter at test time
    pi = make_batch_policy_fn(env, flavor, alpha_sysid)(
        "pi", env.observation_space, env.action_space)

    seeddir = os.path.join(mydir, str(seed))
    ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    assert env.sysid_dim == len(env.sysid_names)
    N = 1000
    x = np.linspace(-4, 4, N)

    test_state = np.array([0.9, 0.9, 0, 0]) # env starting state - upper right corner

    def gen():
        for i, name in enumerate(env.sysid_names):
            sysid = np.zeros((N, env.sysid_dim))
            sysid[:,i] = x
            pi_input = np.hstack([np.tile(test_state, (N, 1)), sysid])
            actions, values = pi.act(False, pi_input)
            actions[actions < -1] = -1
            actions[actions > 1] = 1
            yield name, x, actions

    return list(gen())


# generate data for plotting the mapping from SysID dimension to embedding.
def sysids_to_embeddings(sess, env_id, seed, mydir):

    env = gym.make(env_id)
    assert env.sysid_dim == len(env.sysid_names)

    alpha_sysid = 0 # doesn't matter at test time
    pi = make_batch_policy_fn(env, sysid_batch_policy.EMBED, alpha_sysid)(
        "pi", env.observation_space, env.action_space)

    seeddir = os.path.join(mydir, str(seed))
    ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    if env.sysid_dim == 1:
        N = 1000
        x = np.linspace(*env.sysid_ranges[0], num=N)
        y = pi.sysid_to_embedded(x[:,None])
        return name, x, y

    elif env.sysid_dim == 2:
        N = 256
        x0, x1 = np.meshgrid(
            *(np.linspace(r[0], r[1], N) for r in env.sysid_ranges))
        input = np.concatenate([x0[:,:,None], x1[:,:,None]], axis=2)
        input = input.reshape((-1, 2))
        y = pi.sysid_to_embedded(input)
        y = y.reshape((N, N, 2))
        return env.sysid_ranges, env.sysid_names, y

    else:
        raise NotImplementedError


# print an array of strings surrounded by a box
def boxprint(lines):
    maxlen = max(len(s) for s in lines)
    frame = lambda s, c: c + s + c[::-1]
    rpad = lambda s: s + " " * (maxlen - len(s))
    bar = frame("-" * (maxlen + 2), "  *")
    print(bar)
    for line in lines:
        print(frame(rpad(line), '  | '))
    print(bar)


def set_xterm_title(title):
    sys.stdout.write("\33]0;" + title + "\a")
