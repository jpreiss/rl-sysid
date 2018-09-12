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

    ALPHA = 0.0
    specs = [s for s in lib.multispec_product(spec) if s["alpha_sysid"] == ALPHA]
    n_procs = 6
    n_sysid_samples = 1
    lib.train_multispec(spec, rootdir, n_procs)
    lib.test_multispec(spec, rootdir, n_sysid_samples, n_procs)

    # TODO handle multiple seeds

    embed_dims = list(lib.iter_key(specs, "embed_dim"))
    print("embed_dims:", embed_dims)
    learning_curves = [lib.load_learning_curve(s, rootdir) for s in specs]
    last_k = min(learning_curves[0].shape[0], 5)
    train_final_mean_rews = [np.mean(lc[-last_k:]) for lc in learning_curves]

    results = lib.load_test_results(spec, rootdir)
    results = [(spec, seg) for spec, seg in results if spec["alpha_sysid"] == ALPHA]
    test_rews = [np.mean(list(lib.iter_key(segs, "ep_rews"))) for _, segs in results]

    fig = plt.figure()
    # print(plt.style.available)
    plt.style.use("seaborn-darkgrid")
    plt.plot(embed_dims, train_final_mean_rews, linewidth=2)
    plt.plot(embed_dims, test_rews, linewidth=2)
    plt.xticks(embed_dims)
    plt.xlabel("embedding dimension")
    plt.ylabel("final mean rewards")
    plt.legend(["train", "test"])
    fig.savefig("embed_sweep.pdf")


if __name__ == "__main__":
    main()
