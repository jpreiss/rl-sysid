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

import libexperiment as lib

def test_pickle_path(spec):
    return os.path.join("results", lib.spec_slug(spec), "test_results.pickle")

def train(spec):
    lib.train_all(spec, 4)

def test(spec):
    segs = lib.test_all(spec, 6)
    with open(test_pickle_path(spec), 'wb') as f:
        pickle.dump(segs, f, protocol=4)

def print_test(spec):
    with open(test_pickle_path(spec), 'rb') as f:
        segs = pickle.load(f, encoding="bytes")
    flat_rews = lib.flat_rewards(segs)
    lib.print_test_results(segs, flat_rews)
    lib.print_anova(flat_rews)

def train_vs_test_boxplot(spec):
    with open(test_pickle_path(spec), 'rb') as f:
        segs = pickle.load(f, encoding="bytes")
    last_k = 5
    train_rews = np.array([
        [lib.load_all_learning_curves(spec, flavor, alpha)[:,-last_k].flatten()
            for alpha in spec["alphas"]]
        for flavor in spec["flavors"]
    ])
    test_rews = lib.flat_rewards(segs).reshape(
        (len(spec["flavors"]), len(spec["alphas"]), -1))
    fig = plots.reward_boxplot(spec["flavors"], spec["alphas"], train_rews, test_rews)
    fig.savefig("rewards.pdf")


def learning_curves(spec):
    rews = [lib.load_all_learning_curves(spec, flavor, alpha)
        for flavor, alpha in product(spec["flavors"], spec["alphas"])]
    fig = plots.learning_curves(spec["flavors"], spec["alphas"],
        per_seed_rews=rews, mode="std")
    fig.savefig("learning_curves.pdf")


def main():
    spec = lib.spec_prototype

    train(spec)
    test(spec)
    print_test(spec)
    train_vs_test_boxplot(spec)
    learning_curves(spec)


if __name__ == '__main__':
    main()
