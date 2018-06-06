#!/usr/bin/env python3

import os
import pickle
from itertools import *

import numpy as np
import matplotlib.pyplot as plt

import libexperiment as lib
import plots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def test_pickle_path(spec):
    return os.path.join("results", lib.spec_slug(spec), "test_results.pickle")

def test(spec, n_procs):
    segs = lib.test_all(spec, n_procs)
    with open(test_pickle_path(spec), 'wb') as f:
        pickle.dump(segs, f, protocol=4)

def print_test(spec):
    with open(test_pickle_path(spec), 'rb') as f:
        segs = pickle.load(f, encoding="bytes")
    flat_rews = lib.flat_rewards(segs)
    lib.print_test_results(segs, flat_rews)
    lib.print_anova(flat_rews)
    fig = plots.flat_rew_hist(segs, flat_rews)
    fig.savefig("test_rewards_hist.pdf")

def train_vs_test_boxplot(spec):
    with open(test_pickle_path(spec), 'rb') as f:
        segs = pickle.load(f, encoding="bytes")
    last_k = 10
    train_rews = np.array([
        [lib.load_all_learning_curves(spec, flavor, alpha)[:,-last_k:].flatten()
            for alpha in spec["alphas"]]
        for flavor in spec["flavors"]
    ])
    test_rews = lib.flat_rewards(segs)
    test_rews = np.mean(test_rews, axis=-1).reshape((len(spec["flavors"]), len(spec["alphas"]), -1))
    fig = plots.reward_boxplot(spec["flavors"], spec["alphas"], train_rews, test_rews)
    fig.savefig("rewards.pdf")


def learning_curves(spec):
    params = list(product(spec["flavors"], spec["alphas"]))
    rews = [lib.load_all_learning_curves(spec, flavor, alpha)
        for flavor, alpha in params]
    fig = plots.learning_curves(*zip(*params), per_seed_rews=rews, mode="std")
    fig.savefig("learning_curves.pdf")


def main():
    spec = lib.spec_prototype
    lib.train_all(spec, 4)
    test(spec, 4)
    print_test(spec)
    train_vs_test_boxplot(spec)
    learning_curves(spec)

    if False:
        rews = lib.load_env_mean_rewards(spec,
            *(spec[s][0] for s in ("flavors", "alphas", "seeds")))
        print("env version mean rewards:")
        print(rews)
        plt.hist(rews)
        plt.show()

if __name__ == '__main__':
    main()
