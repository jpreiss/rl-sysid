#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from copy import deepcopy

from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.trpo_mpi import trpo_batch
from baselines.ppo1 import pposgd_batch
import sysid_batch_policy
from sysid_batch_policy import SysIDPolicy, Dim
import baselines.common.tf_util as U
import baselines.common.batch_util as batch
import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from reacher_vectorfield import reacher_vectorfield
import plots

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
            embed = 8,
            agents = env.N,
            window = 20,
        )
        return SysIDPolicy(name=name, flavor=flavor, dim=dim,
            hid_size=64, n_hid=2, alpha_sysid=alpha_sysid)
    return f

def train(sess, env_id, flavor, alpha_sysid, seed, num_timesteps, csvdir):

    set_global_seeds(seed)

    env = gym.make(env_id)
    env.seed(seed)

    policy_fn = make_batch_policy_fn(env, flavor, alpha_sysid)

    gym.logger.setLevel(logging.WARN)

    algo = "ppo"
    if algo == "trpo":
        trained_policy = trpo_batch.learn(env, policy_fn,
            timesteps_per_batch=150, max_timesteps=num_timesteps,
            max_kl=0.01, cg_iters=20, cg_damping=0.1,
            gamma=0.99, lam=0.98,
            vf_iters=4, vf_stepsize=1e-3,
            entcoeff=0.01,
            logdir=csvdir)
    elif algo == "ppo":
        trained_policy = pposgd_batch.learn(env, policy_fn,
            timesteps_per_actorbatch=150,
            max_iters=500,
            clip_param=0.2, entcoeff=0.02,
            optim_epochs=2, optim_stepsize=5e-4, optim_batchsize=512,
            gamma=0.99, lam=0.97, schedule="constant",
            logdir=csvdir
        )
    else:
        assert False, "invalid choice of RL algorithm"
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
    TEST_TIMESTEPS = 150
    TEST_ITERS = 5
    if env_id == "Reacher-Batch-v1":
        assert TEST_TIMESTEPS % 50 == 0

    # while in some cases, you might set stochastic=False at test time
    # to get the "best" actions, for SysID stochasticity could be
    # an important / desired part of the policy
    seg_gen = batch.traj_segment_generator(
        pi, env, TEST_TIMESTEPS, stochastic=True, test=True)
    segs = list(islice(seg_gen, TEST_ITERS))
    return segs

# compute the policy mean action at a fixed task state
# conditioned on the SysID params
def action_conditional(sess, env_id, flavor, seed, mydir):

    assert env_id == "PointMass-Batch-v0"
    env = gym.make(env_id)
    print("action_conditional of:", mydir)

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

        # sanity checks
        if False:
            input = np.tile(np.linspace(0.5, 2, 100)[:,None], (1, 2))
            y = pi.sysid_to_embedded(input_test)
            np.set_printoptions(threshold=np.inf)
            print("test mass = gain:", y)

            input2 = np.column_stack([
                np.linspace(0.5, 2, 100),
                np.linspace(2, 0.5, 100)])
            y = pi.sysid_to_embedded(input2)
            np.set_printoptions(threshold=np.inf)
            print("test mass /= gain:", y)


        y = y.reshape((N, N, 2))
        return env.sysid_ranges, env.sysid_names, y


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
        U.flush_placeholders()
        with tf.Session(graph=g) as sess:
            train(sess, env_id, flavor, alpha_sysid, seed, num_timesteps, csvdir)
            ckpt_path = os.path.join(seeddir, 'trained_model.ckpt')
            saver = tf.train.Saver()
            saver.save(sess, ckpt_path)


def set_xterm_title(title):
    sys.stdout.write("\33]0;" + title + "\a")

def experiment(env_id, n_runs, timesteps, 
    do_train=False,
    do_test=False,
    do_test_results=False,
    do_graph=False,
    do_traces=False,
    do_rew_conditional=False,
    do_action_conditional=False,
    do_embed_mapping=False,
    do_embed_scatters=False,
    do_embed_colors=False,
    ):

    # flavors - defined in policy file, imported
    #alphas = [0.0, 0.1]
    alphas = [0.0, 0.01, 0.03, 0.1]
    seeds = range(n_runs)
    flavs = deepcopy(sysid_batch_policy.flavors)
    n_flav = len(flavs)
    n_alph = len(alphas)

    basedir = "./results_" + env_id.replace("-", "_")
    def dir_fn(basedir, flavor, alpha):
        return os.path.join(basedir, flavor, "alpha" + str(alpha))

    test_pickle_path = os.path.join(basedir, 'test_results.pickle')

    # DEBUG!!
    #flavs = ["embed"]
    #seeds = [3]

    params = list(product(flavs, alphas)) # cartesian product

    if do_train:
        # single-threaded
        for flavor, alpha in params:
            mydir = dir_fn(basedir, flavor, alpha)
            train_one_flavor(env_id, flavor, alpha, seeds, timesteps, mydir)

    #
    # TESTING
    #
    if do_test:
        def test_one(flavor, alpha, seed):
            g = tf.Graph()
            U.flush_placeholders()
            with tf.Session(graph=g) as sess:
                mydir = dir_fn(basedir, flavor, alpha)
                set_xterm_title(mydir)
                return test(sess, env_id, flavor, seed, mydir)

        segs = [
            (flavor, alpha,
                [(seed, test_one(flavor, alpha, seed)) for seed in seeds])
            for flavor, alpha in params
        ]

        with open(test_pickle_path, 'wb') as f:
            pickle.dump(segs, f)

    if do_test_results:
        with open(test_pickle_path, 'rb') as f:
            segs = pickle.load(f)

        def all_flat_rews(seed_segs):
            def flat_rews(segs):
                return flatten_lists(seg["ep_rets"] for seg in segs)
            return [flat_rews(segs) for seed, segs in seed_segs]

        all_rews = np.array([all_flat_rews(seed_segs) for _, _, seed_segs in segs])
        assert all_rews.shape[0] == len(flavs) * len(alphas)
        assert all_rews.shape[1] == len(seeds)

        for (flavor, alpha, seed_segs), rews in zip(segs, all_rews):
            def gen_lines():
                yield "{}, alpha_sysid = {}:".format(flavor, alpha)
                for (seed, _), r in zip(seed_segs, rews):
                    yield " seed {}: mean: {:.2f}, std: {:.2f}".format(
                        seed, np.mean(r), np.std(r))
                yield "overall: mean: {:.2f}, std: {:.2f}".format(
                        np.mean(rews), np.std(rews))
            boxprint(list(gen_lines()))

        # for ANOVA
        each_trial_rews = np.reshape(all_rews, (all_rews.shape[0], -1))
        f, p = sp.stats.f_oneway(*each_trial_rews)
        print("ANOVA each trial results: f = {}, p = {}".format(f, p))

        each_seed_rews = np.mean(all_rews, axis=2)
        f, p = sp.stats.f_oneway(*each_seed_rews)
        print("ANOVA individual seed results: f = {}, p = {}".format(f, p))

        # get mean training reward in last k episodes
        last_k = 2
        def stack_seeds(flavor, alpha):
            def iter_seeds():
                csvdir = dir_fn(basedir, flavor, alpha)
                for seed in seeds:
                    path = os.path.join(csvdir, str(seed), "train_log", "progress.csv")
                    data = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64)
                    rews = data["EpRewMean"]
                    yield rews
            all_rews = np.vstack(iter_seeds())
            return all_rews

        # make LaTeX table
        print()
        print("flavor & $\\alpha$ & training reward mean & test reward mean & test reward std (per-episode) & test reward std (per-seed) \\\\")
        for (flavor, alpha), pertrial, perseed in zip(
            product(flavs, alphas), each_trial_rews, each_seed_rews):
            training_rews = stack_seeds(flavor, alpha)
            #print(flavor, alpha)
            #for i, rews in enumerate(training_rews):
                #print("seed {}: last {} training rews = {}".format(
                    #i, last_k, np.mean(rews[-last_k:])))
            training_reward = np.mean(training_rews[:,-last_k:])
            print("{} & {} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(flavor, alpha,
                training_reward, np.mean(pertrial), np.std(pertrial), np.std(perseed)))

        train_rews = np.array([
            [stack_seeds(flavor, alpha)[:,-last_k:].flatten() for alpha in alphas]
            for flavor in flavs
        ])
        test_rews = all_rews.reshape((len(flavs), len(alphas), -1))
        fig = plots.reward_boxplot(flavs, alphas, train_rews, test_rews)
        fig.savefig("rewards.pdf")



    if do_embed_scatters:
        with open(test_pickle_path, 'rb') as f:
            segs = pickle.load(f)

        def get_scatter_data(segs):
            def flatten(key):
                for seg in segs:
                    yield seg[key]
            trues = np.vstack(flatten("embed_true"))
            estimates = np.vstack(flatten("embed_estimate"))
            assert estimates.shape == trues.shape
            N, dim = trues.shape
            trues = trues[::100,:]
            estimates = estimates[::100,:]
            if trues.shape[1] > 1:
                trues = trues[:,0]
                estimates = estimates[:,0]
            return trues.flatten(), estimates.flatten()

        def datas(segs):
            for flavor, alpha, seed_segs in segs:
                skip_flavors = [sysid_batch_policy.BLIND, sysid_batch_policy.EXTRA]
                if flavor in skip_flavors or alpha != 0.1:
                    continue
                seed, segs = seed_segs[0]
                ac, emb = get_scatter_data(segs)
                yield ac, emb, flavor

        trues, estimates, flavors = zip(*datas(segs))
        fig = plots.embed_scatter(np.row_stack(trues), np.row_stack(estimates),
            list(flavors), ["gain (actual)", "embedding (actual)"])
        fig.savefig("embed_scatter.pdf")

    # TODO extract from segs
    #if do_rew_conditional:
        #with open(test_pickle_path, 'rb') as f:
            #segs = pickle.load(f)
        #for flav_rews, flav_sysids in zip(rews, sysids):
            #r = flatten_lists(flav_rews.flat)
            #s = flatten_lists(flav_sysids.flat)
            #plt.scatter(s, r)
            #plt.xlabel("gain")
            #plt.ylabel("rewards")
            #plt.show()

    if do_action_conditional:
        for flavor, alpha, seed in product(flavs, alphas, seeds):
            g = tf.Graph()
            U.flush_placeholders()
            with tf.Session(graph=g) as sess:
                mydir = dir_fn(basedir, flavor, alpha)
                set_xterm_title(mydir)
                graphs = action_conditional(sess, env_id, flavor, seed, mydir)
            sysid_dim = len(graphs)
            for name, x, action in graphs:
                _, act_dim = action.shape
                plt.clf()
                plt.title("SysID dimension: " + name)
                for i in range(act_dim):
                    plt.subplot(1, act_dim, i+1)
                    plt.plot(x, action[:,i])
                    plt.xlabel(name)
                    plt.ylabel("action[{}]".format(i))
                    plt.grid(True)
                    plt.ylim([-1.5, 1.5])
            plt.show()

    if do_embed_mapping:
        g = tf.Graph()
        U.flush_placeholders()
        with tf.Session(graph=g) as sess:
            mydir = dir_fn(basedir, sysid_batch_policy.EMBED, 0.1)
            set_xterm_title(mydir)
            seed = 1
            mappings = sysids_to_embeddings(sess, env_id, seed, mydir)
        sysid_dim = len(mappings)
        for name, sysid, embed in mappings:
            _, embed_dim = embed.shape
            plt.clf()
            plt.title("SysID dimension: " + name)
            for i in range(embed_dim):
                plt.subplot(1, embed_dim, i+1)
                plt.plot(sysid, embed[:,i])
                plt.xlabel(name)
                plt.ylabel("embed[{}]".format(i))
            plt.show()

    if do_embed_colors:
        g = tf.Graph()
        U.flush_placeholders()
        with tf.Session(graph=g) as sess:
            mydir = dir_fn(basedir, sysid_batch_policy.EMBED, 0.1)
            set_xterm_title(mydir)
            seed = 0
            (xrange, yrange), names, mappings = sysids_to_embeddings(
                sess, env_id, seed, mydir)
        #mappings[:,:,0] = mappings[:,:,1]
        assert len(mappings.shape) == 3
        assert mappings.shape[2] == 2
        fig = plots.embed2d_color(mappings, xrange, yrange, names)
        fig.savefig("embed_colors.pdf")
        print("saved the figure")


    if do_graph:
        def stack_seeds(flavor, alpha):
            def iter_seeds():
                csvdir = dir_fn(basedir, flavor, alpha)
                for seed in seeds:
                    path = os.path.join(csvdir, str(seed), "train_log", "progress.csv")
                    data = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64)
                    rews = data["EpRewMean"]
                    assert len(rews.shape) == 1
                    print("iter seeds shape:", rews.shape)
                    yield rews
            all_rews = np.vstack(iter_seeds())
            return all_rews
        plt.clf()
        plt.hold(True)
        color_list = plt.cm.Set3(np.linspace(0, 1, len(params)))
        for (flavor, alpha), color in zip(params, color_list):
            rews = stack_seeds(flavor, alpha)
            #rews = rews[1][None,:]
            for seed_rew in rews:
                smooth = 7
                seed_rew = np.convolve(seed_rew, 1.0 / smooth * np.ones(smooth), mode="valid")
                plt.plot(seed_rew, color=color, linewidth=2, 
                    label=str((flavor, alpha)))
        #plt.yticks(np.arange(-900, -150, 25))
        plt.yticks(np.arange(-200, -50, 10))
        plt.grid(True, axis="y")
        plt.xlabel('iteration')
        plt.ylabel('mean reward per episode')
        plt.legend(loc='lower right')
        plt.show()


    if do_traces:
        n = len(list(product(flavs, alphas)))
        for i, (flavor, alpha) in enumerate(product(flavs, alphas)):
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
    num_timesteps = 150 * 32 * 300
    n_runs = 3
    experiment("HalfCheetah-Batch-v1", n_runs, num_timesteps,
        do_train        = True,
        do_test         = True,
        do_test_results = True,
        do_graph        = True,
        do_traces       = False,
        do_rew_conditional = False,
        do_action_conditional = False,
        do_embed_scatters = False,
        do_embed_mapping = False,
        do_embed_colors = False,
    )

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

if __name__ == '__main__':
    main()
