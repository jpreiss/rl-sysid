import itertools as it
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import libexperiment as lib
import plots

def main():
    rootdir, _ = os.path.split(__file__)
    spec_path = os.path.join(rootdir, "config.json")
    with open(spec_path) as f:
        spec = lib.Spec(json.load(f))

    specs = [s for s in lib.multispec_product(spec)]

    n_procs = 5
    n_sysid_samples = 4
    lib.train_multispec(spec, rootdir, n_procs)
    lib.test_multispec(spec, rootdir, n_sysid_samples, n_procs)

    learning_curves = [lib.load_learning_curve(s, rootdir) for s in specs]
    last_k = min(learning_curves[0].shape[0], 5)
    train_rews = [lc[-last_k:].flatten() for lc in learning_curves]

    results = lib.load_test_results(spec, rootdir)
    test_rews = [np.array(list(lib.iter_key(segs, "ep_rews"))).flatten() for _, segs in results]

    fig = plt.figure(figsize=(6,10))
    plt.style.use("seaborn-dark")

    keys = ["flavor", "alpha_sysid"]
    (flavors, alphas), train_table = lib.tabulate(specs, train_rews, keys)
    _, test_table = lib.tabulate(specs, test_rews, keys)

    # collapse seed axis
    train_table = lib.flatlast(train_table)
    test_table = lib.flatlast(test_table)

    # nested boxplot expects 3 levels of nesting
    labels = [flavors, alphas, ["train", "test"]]
    fig = plots.nested_boxplot(labels, train_table, test_table, aspect_ratio=1.2)
    fig.savefig("embed_fix.pdf")


if __name__ == "__main__":
    main()
