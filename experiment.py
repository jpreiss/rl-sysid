#!/usr/bin/env python3

import os
import pickle
from itertools import *
import pdb

import numpy as np
import matplotlib.pyplot as plt

import libexperiment as lib
import plots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def test_result_path(spec):
    return os.path.join("results", lib.spec_slug(spec))

def test_pickle_path(spec):
    return os.path.join(test_result_path(spec), "test_results.pickle")

def test(spec, sysid_iters, n_procs):
    segs = lib.test_all(spec, sysid_iters, n_procs)
    with open(test_pickle_path(spec), 'wb') as f:
        pickle.dump(segs, f, protocol=4)

def print_test(spec):
    with open(test_pickle_path(spec), 'rb') as f:
        segs = pickle.load(f, encoding="bytes")
    flat_rews = lib.flat_rewards(segs)
    lib.print_test_results(segs, flat_rews)
    lib.print_anova(flat_rews)
    fig = plots.flat_rew_hist(segs, flat_rews)
    fig.savefig(os.path.join(test_result_path(spec), "test_rewards_hist.pdf"))

def train_vs_test_boxplot(spec):
    with open(test_pickle_path(spec), 'rb') as f:
        segs = pickle.load(f, encoding="bytes")
    last_k = 12
    train_rews = np.array([
        [lib.load_all_learning_curves(spec, flavor, alpha)[:,-last_k:,:]
            for alpha in spec["alphas"]]
        for flavor in spec["flavors"]
    ])
    test_rews = lib.flat_rewards(segs)

    nF, nA = train_rews.shape[:2]
    train_rews = train_rews.reshape((nF, nA, -1))
    test_rews = test_rews.reshape((nF, nA, -1))

    fig = plots.reward_boxplot(spec["flavors"], spec["alphas"], train_rews, test_rews)
    fig.savefig(os.path.join(test_result_path(spec), "rewards.pdf"))


def learning_curves(spec):
    flavs, alphs = spec["flavors"], spec["alphas"]
    rews = np.stack(np.stack(lib.load_all_learning_curves(spec, flavor, alpha)
            for alpha in alphs)
        for flavor in flavs)
    fig = plots.learning_curves(flavs, alphs, rews, mode="bound")
    fig.savefig(os.path.join(test_result_path(spec), "learning_curves.pdf"))


def sysid_err_over_episodes(spec):
    with open(test_pickle_path(spec), 'rb') as f:
        segs = pickle.load(f, encoding="bytes")
    segs = lib.SegArray(segs)
    def f(flavor, alpha, iseed, ienv):
        seed, seg = segs.find(flavor, alpha)[iseed]
        seg = seg[0]
        #assert type(seg) == type({})
        true = seg["embed_true"][ienv,:]
        est = seg["embed_estimate"][:,ienv,:]
        err = np.mean((true - est)**2, axis=1)
        plt.plot(err)

    def g(flavor, alpha):
        for i in range(20):
            f(flavor, alpha, 0, i)


def embed_hist(spec):

    with open(test_pickle_path(spec), 'rb') as f:
        all_segs = lib.SegArray(pickle.load(f, encoding="bytes"))
    seed, segs = all_segs.find("embed", 0.1)[0]
    embeds = np.stack(lib.iter_key(segs, "embed_true"))
    runs, N, dim = embeds.shape
    embeds = embeds.reshape((-1, dim))
    n_row = int(np.floor(np.sqrt(dim)))
    n_col = int(np.ceil(dim / float(n_row)))
    for i in range(dim):
        plt.subplot(n_row, n_col, i + 1)
        plt.hist(embeds[:,i])
    #plt.show()

def main():
    spec = lib.spec_prototype
    results_path = test_result_path(spec)
    #spec["seeds"] = [0,1,2]
    #spec["alphas"] = [0.0, 0.005]
    #spec["flavors"] = ["blind", "extra", "embed"]

    #sysid_err_over_episodes(spec)

    #lib.train_all(spec, 4)
    #test(spec, sysid_iters=4, n_procs=1)
    print_test(spec)
    #train_vs_test_boxplot(spec)
    #learning_curves(spec)
    #embed_hist(spec)

    # why do I need to chdir???
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, "results"))
    os.remove("last")
    os.symlink(lib.spec_slug(spec), "last")
    os.chdir(cwd)

    if False:
        rews = lib.load_env_mean_rewards(spec,
            *(spec[s][0] for s in ("flavors", "alphas", "seeds")))
        print("env version mean rewards:")
        print(rews)
        plt.hist(rews)

if __name__ == '__main__':
    main()
