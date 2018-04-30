#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.trpo_mpi import trpo_batch
from sysid_batch_policy import SysIDPolicy, Dim, flavors
import baselines.common.tf_util as U
import baselines.common.batch_util as batch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from reacher_vectorfield import reacher_vectorfield

import csv
import glob
import multiprocessing
import pickle
import sys
import time
from itertools import *

def make_batch_policy_fn(env, flavor, alpha_sysid):
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
            embed = 2,
            agents = env.N,
            window = 20,
        )
        return SysIDPolicy(name=name, flavor=flavor, dim=dim,
            hid_size=32, n_hid=2, alpha_sysid=alpha_sysid)
    return f

def train(sess, env_id, flavor, alpha_sysid, seed, num_timesteps, csvdir):

    set_global_seeds(seed)

    env = gym.make(env_id)
    env.seed(seed)

    policy_fn = make_batch_policy_fn(env, flavor, alpha_sysid)

    gym.logger.setLevel(logging.WARN)

    trained_policy = trpo_batch.learn(env, policy_fn,
        timesteps_per_batch=256, max_timesteps=num_timesteps,
        max_kl=0.01, cg_iters=10, cg_damping=0.1,
        gamma=0.99, lam=0.98,
        vf_iters=2, vf_stepsize=1e-3,
        entcoeff=0.01,
        logdir=csvdir)
    env.close()

def test(sess, env_id, flavor, seed, mydir):

    set_global_seeds(100+seed)

    env = gym.make(env_id)
    env.seed(100+seed)

    alpha_sysid = 0 # doesn't matter at test time
    pi = make_batch_policy_fn(env, flavor, alpha_sysid)(
        "pi", env.observation_space, env.action_space)

    seeddir = os.path.join(mydir, str(seed))
    ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    # many short iters: denser sampling of true SysID values
    TEST_TIMESTEPS = 200
    TEST_ITERS = 1
    if env_id == "Reacher-Batch-v1":
        assert TEST_TIMESTEPS % 50 == 0
    rews = []

    # while in some cases, you might set stochastic=False at test time
    # to get the "best" actions, for SysID stochasticity could be
    # an important / desired part of the policy
    seg_gen = batch.traj_segment_generator(
        pi, env, TEST_TIMESTEPS, stochastic=True, test=True)
    for seg in islice(seg_gen, TEST_ITERS):
        these_rews = seg["ep_rets"]
        rews.extend(these_rews)

    return rews


def make_vectorfield(sess, env_id, flavor, seed, mydir):

    set_global_seeds(100+seed)

    env = gym.make(env_id)
    env.seed(100+seed)

    alpha_sysid = 0 # doesn't matter at test time
    pi = make_batch_policy_fn(env, flavor, alpha_sysid)(
        "pi", env.observation_space, env.action_space)

    seeddir = os.path.join(mydir, str(seed))
    ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    reacher_vectorfield(env, pi)


def train_one_flavor(env_id, flavor, alpha_sysid, seeds, num_timesteps, mydir):
    os.makedirs(mydir, exist_ok=True)
    for seed in seeds:
        seeddir = os.path.join(mydir, str(seed))
        csvdir = os.path.join(seeddir, 'train_log')
        os.makedirs(csvdir, exist_ok=True)

        g = tf.Graph()
        U.flush_placeholders()
        with tf.Session(graph=g) as sess:
            train(sess, env_id, flavor, alpha_sysid, seed, num_timesteps, csvdir)
            ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
            saver = tf.train.Saver()
            saver.save(sess, ckpt_path)


def set_xterm_title(title):
    sys.stdout.write("\33]0;" + title + "\a")

def experiment(env_id, n_runs, timesteps, do_train=False, do_test=False, do_test_results=False, do_graph=False, do_traces=False):

    # flavors - defined in policy file, imported
    #alphas = [0, 0.1]
    alphas = [0.1]
    seeds = range(n_runs)
    n_flav = len(flavors)
    n_alph = len(alphas)

    basedir = "./results_" + env_id.replace("-", "_")
    def dir_fn(basedir, flavor, alpha):
        return os.path.join(basedir, flavor, "alpha" + str(alpha))

    test_pickle_path = os.path.join(basedir, 'test_results.pickle')

    params = list(product(flavors, alphas)) # cartesian product

    if do_train:
        # single-threaded
        for flavor, alpha in params:
            mydir = dir_fn(basedir, flavor, alpha)
            set_xterm_title(mydir)
            train_one_flavor(env_id, flavor, alpha, seeds, timesteps, mydir)

    #
    # TESTING
    #
    if do_test:
        # will fill arrays, possibly ragged
        rews = np.full((n_flav, n_alph, n_runs), None)
        for (i, flavor), (j, alpha), seed in product(
            enumerate(flavors), enumerate(alphas), seeds):
            g = tf.Graph()
            U.flush_placeholders()
            with tf.Session(graph=g) as sess:
                mydir = dir_fn(basedir, flavor, alpha)
                set_xterm_title(mydir)
                rews[i,j,seed] = test(sess, env_id, flavor, seed, mydir)

        with open(test_pickle_path, 'wb') as f:
            pickle.dump(rews, f)

    if do_test_results:
        with open(test_pickle_path, 'rb') as f:
            rews = pickle.load(f)

        def seed_line(i, j, seed):
            return " seed {}: mean: {:.2f}, std: {:.2f}".format(
                seed, np.mean(rews[i,j,seed]), np.std(rews[i,j,seed]))
        def flavor_lines(i, j):
            yield "{}, alpha_sysid = {}:".format(flavors[i], alphas[j])
            yield from (seed_line(i, j, seed) for seed in seeds)
            mean_all = np.mean(np.concatenate(rews[i,j,:]))
            std_all = np.std(np.concatenate(rews[i,j,:]))
            yield "overall: mean: {:.2f}, std: {:.2f}".format(mean_all, std_all)
            yield ""
        boxprint(list(chain.from_iterable(
            flavor_lines(i, j) for i, j in product(range(n_flav), range(n_alph))))) # lisp??

    if do_graph:
        def stack_seeds(flavor, alpha):
            def iter_seeds():
                csvdir = dir_fn(basedir, flavor, alpha)
                for seed in os.listdir(csvdir):
                    path = os.path.join(csvdir, seed, "train_log", "progress.csv")
                    data = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64)
                    rews = data["EpRewMean"]
                    assert len(rews.shape) == 1
                    yield rews
            all_rews = np.vstack(iter_seeds())
            return all_rews
        plt.clf()
        plt.hold(True)
        color_list = plt.cm.Set3(np.linspace(0, 1, len(params)))
        for (flavor, alpha), color in zip(params, color_list):
            rews = stack_seeds(flavor, alpha)
            for seed_rew in rews:
                seed_rew = np.convolve(seed_rew, np.ones(5), mode="valid")
                plt.plot(seed_rew, color=color, linewidth=2, 
                    label=str((flavor, alpha)))
        plt.xlabel('iteration')
        plt.ylabel('mean reward per episode')
        plt.legend(loc='lower right')
        plt.show()


    if do_traces:
        n = len(list(product(flavors, alphas)))
        for i, (flavor, alpha) in enumerate(product(flavors, alphas)):
            seed = seeds[0]
            g = tf.Graph()
            U.flush_placeholders()
            with tf.Session(graph=g) as sess:
                mydir = dir_fn(basedir, flavor, alpha)
                set_xterm_title(mydir)
                plt.subplot(1,n,i+1)
                plt.title(mydir)
                make_vectorfield(sess, env_id, flavor, seed, mydir)
        plt.show()


def boxprint(lines):
    maxlen = max(len(s) for s in lines)
    frame = lambda s, c: c + s + c[::-1]
    rpad = lambda s: s + " " * (maxlen - len(s))
    bar = frame("-" * (maxlen + 2), "  *")
    print(bar)
    for line in lines:
        print(frame(rpad(line), '  | '))
    print(bar)

def main():
    num_timesteps = 150 * 32 * 200
    n_runs = 1
    experiment("PointMass-Batch-v0", n_runs, num_timesteps,
        do_train        = False,
        do_test         = False,
        do_test_results = False,
        do_graph        = True,
        do_traces       = False,
    )

if __name__ == '__main__':
    main()
