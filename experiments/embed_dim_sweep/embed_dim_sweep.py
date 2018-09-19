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

    specs = [s for s in lib.multispec_product(spec)]
    n_procs = 6
    n_sysid_samples = 1
    #lib.train_multispec(spec, rootdir, n_procs)
    #lib.test_multispec(spec, rootdir, n_sysid_samples, n_procs)

    learning_curves = [lib.load_learning_curve(s, rootdir) for s in specs]
    last_k = min(learning_curves[0].shape[0], 5)
    train_rews = [np.mean(lc[-last_k:]) for lc in learning_curves]

    results = lib.load_test_results(spec, rootdir)
    test_rews = [np.mean(list(lib.iter_key(segs, "ep_rews"))) for _, segs in results]

    tups = list(zip(specs, train_rews, test_rews))
    alphas = spec["alpha_sysid"]

    fig = plt.figure(figsize=(6,10))
    plt.style.use("seaborn-darkgrid")

    import pdb; pdb.set_trace()
    for i, alpha in enumerate(alphas):
        print(f"doing alpha={alpha}")
        plt.subplot(len(alphas), 1, i + 1)
        filtered = [tup for tup in tups if tup[0]["alpha_sysid"] == alpha]
        # averaging over seeds
        specs_f, train_rews_f, test_rews_f = zip(*filtered)
        handles = []
        for rews in (train_rews_f, test_rews_f):
            cores, mins = zip(*lib.group_reduce(specs_f, rews, np.amin))
            maxes = [r for spec, r in lib.group_reduce(specs_f, rews, np.amax)]
            means = [r for spec, r in lib.group_reduce(specs_f, rews, np.mean)]
            embed_dims = list(lib.iter_key(cores, "embed_dim"))
            handles.append(plt.fill_between(embed_dims, mins, maxes, alpha=0.3))
            plt.plot(embed_dims, means)

        plt.xticks(embed_dims)
        plt.xlabel("embedding dimension")
        plt.ylabel("final mean rewards")
        plt.legend(handles, ["train", "test"])
    fig.savefig("embed_sweep.pdf")


if __name__ == "__main__":
    main()
