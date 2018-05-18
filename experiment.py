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

from libexperiment import *


def experiment(env_id, n_runs, n_iters,
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
    alphas = [0.00, 0.01]
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

    # local fn to spread computation.
    # fn should take flavor, alpha
    def grid(fn, arg_fn, procs, debug_singlethread=False):

        if debug_singlethread:
            return [fn(*arg_fn(flavor, alpha)) for flavor, alpha in params]
        else:
            asyncs = []
            pool = multiprocessing.Pool(processes=procs)
            for flavor, alpha in params:
                def callback(arg):
                    print("{} / alpha = {} done".format(flavor, alpha))
                args = arg_fn(flavor, alpha)
                res = pool.apply_async(fn, args, callback=callback)
                asyncs.append(res)
            results = [res.get() for res in asyncs]
            return results


    if do_train:
        multiproc = True
        if multiproc:
            def train_arg_fn(flavor, alpha):
                mydir = dir_fn(basedir, flavor, alpha)
                return env_id, flavor, alpha, seeds, n_iters, mydir
            grid(train_one_flavor, train_arg_fn, procs=5)
        else:
            # single-threaded
            for flavor, alpha in params:
                mydir = dir_fn(basedir, flavor, alpha)
                train_one_flavor(env_id, flavor, alpha, seeds, n_iters, mydir)

    #
    # TESTING
    #
    if do_test:

        def test_arg_fn(flavor, alpha):
            mydir = dir_fn(basedir, flavor, alpha)
            return (env_id, flavor, alpha, seeds, mydir)

        segs = grid(test_one_flavor, test_arg_fn, procs=6)

        with open(test_pickle_path, 'wb') as f:
            pickle.dump(segs, f)


    if do_test_results:
        print("loading pickle")
        with open(test_pickle_path, 'rb') as f:
            segs = pickle.load(f)

        def all_flat_rews(seed_segs):
            def flat_rews(segs):
                return flatten_lists(seg["ep_rews"] for seg in segs)
            return [flat_rews(segs) for seed, segs in seed_segs]

        print("processing pickle")
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

        print("doing ANOVA")
        # for ANOVA
        each_trial_rews = np.reshape(all_rews, (all_rews.shape[0], -1))
        f, p = sp.stats.f_oneway(*each_trial_rews)
        print("ANOVA each trial results: f = {}, p = {}".format(f, p))

        each_seed_rews = np.mean(all_rews, axis=2)
        f, p = sp.stats.f_oneway(*each_seed_rews)
        print("ANOVA individual seed results: f = {}, p = {}".format(f, p))

        # get mean training reward in last k episodes
        last_k = 5
        def stack_seeds(flavor, alpha):
            def iter_seeds():
                csvdir = dir_fn(basedir, flavor, alpha)
                for seed in seeds:
                    path = os.path.join(csvdir, str(seed), "train_log", "progress.csv")
                    data = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64)
                    rews = data["EpRewMean"]
                    yield rews
            all_rews = list(iter_seeds())
            return np.vstack(all_rews)

        print("making latex table")
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
        rews = [load_seed_rews(dir_fn(basedir, flavor, alpha), seeds)
            for flavor, alpha in params]
        fig = plots.learning_curves(*zip(*params), per_seed_rews=rews, mode="std")
        fig.savefig("learning_curves.pdf")
        print("saved learning curves plot.")


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


def main():
    env = "HalfCheetah-Batch-v1"
    n_runs = 5
    n_iters = 400
    experiment("HalfCheetah-Batch-v1", n_runs, n_iters,
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

if __name__ == '__main__':
    main()
