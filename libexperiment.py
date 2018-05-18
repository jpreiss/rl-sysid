#!/usr/bin/env python3

from copy import deepcopy
from itertools import *
import os
import sys

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


import sysid_batch_policy
from sysid_batch_policy import SysIDPolicy, Dim


def make_batch_policy_fn(env, flavor, alpha_sysid):
    embed_dim = 6 # TODO: make parameter
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
            embed = embed_dim,
            agents = env.N,
            window = 8,
        )
        return SysIDPolicy(name=name, flavor=flavor, dim=dim,
            hid_size=48, n_hid=2, alpha_sysid=alpha_sysid)
    return f


# applies the known good hyperparameters, etc.
def train(sess, env_id, flavor, alpha_sysid, seed, num_iters, csvdir):

    set_global_seeds(seed)

    env = gym.make(env_id)
    env.seed(seed)

    policy_fn = make_batch_policy_fn(env, flavor, alpha_sysid)

    gym.logger.setLevel(logging.WARN)

    algo = "ppo"
    if algo == "trpo":
        trained_policy = trpo_batch.learn(env, policy_fn,
            timesteps_per_batch=150, max_iters=num_iters,
            max_kl=0.01, cg_iters=20, cg_damping=0.1,
            gamma=0.99, lam=0.98,
            vf_iters=4, vf_stepsize=1e-3,
            entcoeff=0.015,
            logdir=csvdir)
    elif algo == "ppo":
        trained_policy = pposgd_batch.learn(env, policy_fn,
            timesteps_per_actorbatch=150,
            max_iters=num_iters,
            clip_param=0.2, entcoeff=0.01,
            optim_epochs=2, optim_stepsize=1e-3, optim_batchsize=256,
            gamma=0.99, lam=0.96, schedule="constant",
            logdir=csvdir
        )
    else:
        assert False, "invalid choice of RL algorithm"
    env.close()


# load the policy and test, return array of "seg" dictionaries
def test(sess, env, flavor, seed, mydir):

    set_global_seeds(100+seed)
    env.seed(100+seed)

    alpha_sysid = 0 # doesn't matter at test time
    pi = make_batch_policy_fn(env, flavor, alpha_sysid)(
        "pi", env.observation_space, env.action_space)

    seeddir = os.path.join(mydir, str(seed))
    ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    # while in some cases, you might set stochastic=False at test time
    # to get the "best" actions, for SysID stochasticity could be
    # an important / desired part of the policy
    seg_gen = batch2.sysid_simple_generator(pi, env, stochastic=True, test=True)
    TEST_ITERS = 5
    segs = list(islice(seg_gen, TEST_ITERS))
    return segs


# train the policy, saving the training logs and trained policy
def train_one_flavor(env_id, flavor, alpha_sysid, seeds, iters, mydir):
    os.makedirs(mydir, exist_ok=True)
    for i, seed in enumerate(seeds):
        print("training {}, alpha = {}, seed = {} ({} out of {} seeds)".format(
            flavor, alpha_sysid, seed, i + 1, len(seeds)))
        status = "{} | alpha = {} | seed {}/{}".format(
            flavor, alpha_sysid, i + 1, len(seeds))
        set_xterm_title(status)
        seeddir = os.path.join(mydir, str(seed))
        csvdir = os.path.join(seeddir, 'train_log')
        os.makedirs(csvdir, exist_ok=True)

        g = tf.Graph()
        U.flush_placeholders() # TODO get rid of U
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=g, config=config) as sess:
            train(sess, env_id, flavor, alpha_sysid, seed, iters, csvdir)
            ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
            saver = tf.train.Saver()
            saver.save(sess, ckpt_path)


def test_one_flavor(env_id, flavor, alpha, seeds, mydir):
    env = gym.make(env_id)

    def test_one(flavor, alpha, seed):
        env.seed(100+seed)
        g = tf.Graph()
        U.flush_placeholders()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=g, config=config) as sess:
            return test(sess, env, flavor, seed, mydir)

    return (flavor, alpha,
        [(seed, test_one(flavor, alpha, seed)) for seed in seeds])

# return the reward learning curves for all seeds
# stacked into a (n_seeds, training_iters) array
def load_seed_rews(csvdir, seeds):
    def iter_seeds():
        for seed in seeds:
            path = os.path.join(csvdir, str(seed), "train_log", "progress.csv")
            data = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64)
            rews = data["EpRewMean"]
            assert len(rews.shape) == 1
            yield rews
    return np.vstack(iter_seeds())

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


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def set_xterm_title(title):
    sys.stdout.write("\33]0;" + title + "\a")
