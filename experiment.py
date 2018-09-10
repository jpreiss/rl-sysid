#!/usr/bin/env python3

from itertools import *
import os
import pdb
import pickle
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt

import libexperiment as lib
import plots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def print_test(multispec, rootdir):
    results = lib.load_test_results(multispec, rootdir)
    lib.print_test_results(multispec, results)
    # TODO multispec plots
    #lib.print_anova(flat_rews)
    #fig = plots.flat_rew_hist(segs, flat_rews)
    #p, _ = lib.find_spec_dir(spec)
    #fig.savefig(os.path.join(p, "test_rewards_hist.pdf"))

def train_vs_test_boxplot(spec):
    raise NotImplementedError("multispec")
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

    for i, f in enumerate(spec["flavors"]):
        for j, a in enumerate(spec["alphas"]):
            print("policy:", f, a)
            tr = train_rews[i,j].flatten()
            te = test_rews[i,j].flatten()
            print("{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}".format(
                np.mean(tr), np.std(tr), np.mean(te), np.std(te), np.mean(te) - np.mean(tr)))

    fig = plots.reward_boxplot(spec["flavors"], spec["alphas"], train_rews, test_rews)
    p, _ = lib.find_spec_dir(spec)
    fig.savefig(os.path.join(p, "rewards.pdf"))

"""
def latex_table(spec):
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

    with open("reward_table.tex") as f:
        f.write("train mu, train sigma, test mu, test sigma, generalization error\n"
        for i in range(nF):
            for j in range(nA):
                if spec["flavors"][i] == "blind" and spec["alphas"][j] != 0.0:
                    continue
                trmu = np.mean(train_rews[i,j])
                trstd = np.std(train_rews[i,j])
                temu = np.mean(test_rews[i,j])
                testd = np.std(test_rews[i,j])
                gerr = temu - trmu
                f.write("{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}\n".format(
                    trmu, trstd, temu, testd, gerr))
"""


def learning_curves(spec):
    raise NotImplementedError("multispec")
    flavs, alphs = spec["flavors"], spec["alphas"]
    #rews = np.stack(np.stack(lib.load_all_learning_curves(spec, flavor, alpha)
            #for alpha in alphs)
        #for flavor in flavs)
    rews = [[lib.load_all_learning_curves(spec, flavor, alpha)
            for alpha in alphs]
        for flavor in flavs]
    min_iters = min(lc.shape[1] for lc in chain.from_iterable(rews))
    rews = np.stack(np.stack(fa[:,:min_iters,:] for fa in f) for f in rews)

    fig = plots.learning_curves(flavs, alphs, rews, mode="std")
    p, _ = lib.find_spec_dir(spec)
    fig.savefig(os.path.join(p, "learning_curves.pdf"))


def sysid_err_over_episodes(spec):
    raise NotImplementedError("multispec")
    with open(test_pickle_path(spec), 'rb') as f:
        segs = pickle.load(f, encoding="bytes")
    segs = lib.SegArray(segs)
    def f(flavor, alpha, iseed, ienv):
        seed, seg = segs.find(flavor, alpha)[iseed]
        seg = seg[0]
        #assert type(seg) == type({})
        print(seg.keys())
        true = seg["est_true"][ienv,:]
        est = seg["est"][:,ienv,:]
        err = np.mean((true - est)**2, axis=1)
        plt.plot(err)

    def g(flavor, alpha):
        for i in range(20):
            f(flavor, alpha, 0, i)


def embed_hist(spec):
    raise NotImplementedError("multispec")
    with open(test_pickle_path(spec), 'rb') as f:
        all_segs = lib.SegArray(pickle.load(f, encoding="bytes"))
    seed, segs = all_segs.find("embed")[0]
    try:
        embeds = np.stack(lib.iter_key(segs, "est_true"))
        runs, N, dim = embeds.shape
        embeds = embeds.reshape((-1, dim))
        n_row = int(np.floor(np.sqrt(dim)))
        n_col = int(np.ceil(dim / float(n_row)))
        for i in range(dim):
            plt.subplot(n_row, n_col, i + 1)
            plt.hist(embeds[:,i])
        p, _ = lib.find_spec_dir(spec)
        plt.gcf().savefig(os.path.join(p, "embed_hist.pdf"))
    except KeyError:
        print("experiment had no embed-flavoragents. embed_hist.pdf not written.")


def main():
    spec = lib.multispec
    rootdir = lib.check_spec_dir(spec, "./results")

    np.seterr(all="raise")

    n_procs = 1
    lib.train_multispec(spec, rootdir, n_procs)
    lib.test_multispec(spec, rootdir, n_sysid_samples=8, n_procs=n_procs)
    print_test(spec, rootdir)
    #train_vs_test_boxplot(spec)
    #learning_curves(spec)
    #embed_hist(spec)

    if os.path.exists("last"):
        os.remove("last")
    os.symlink(os.path.relpath(rootdir), "last")


if __name__ == '__main__':
    main()
