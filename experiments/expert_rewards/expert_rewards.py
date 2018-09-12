import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import libexperiment as lib


def main():
    rootdir, _ = os.path.split(__file__)
    spec_path = os.path.join(rootdir, "config.json")
    with open(spec_path) as f:
        spec = lib.Spec(json.load(f))

    # TEMP testing
    spec["train_iters"] = 2

    # n_seeds == n of environments tested
    n_seeds = 4
    spec["seed"] = list(1000 + np.arange(n_seeds))
    specs = lib.multispec_product(spec)
    lib.train_multispec(spec, rootdir, n_procs)

    learning_curves = [lib.load_learning_curve(s, rootdir) for s in specs]
    last_k = min(learning_curves[0].shape[0], 5)
    train_final_mean_rews = [np.mean(lc[-last_k:]) for lc in learning_curves]

    fig = plt.figure()
    plt.style.use("seaborn-darkgrid")
    plt.hist(train_final_mean_rews)
    plt.xlabel("training reward")
    plt.ylabel("frequency")



if __name__ == "__main__":
    main()
