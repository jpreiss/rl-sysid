import argparse
import itertools as it
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import libexperiment as lib
import plots

def main(rootdir, outpath, condense_seed=True, transpose=False):

    spec_path = os.path.join(rootdir, "config.json")
    with open(spec_path) as f:
        multispec = lib.Spec(json.load(f))
    rootdir = os.path.join(rootdir, "results")
    specs = [s for s in lib.multispec_product(multispec)]

    multis = multispec.multi_items()
    if "seed" in multis:
        del multis["seed"]
    items = sorted(multis.items())
    if transpose:
        items = reversed(items)
    keys, multivals = zip(*items)
    if len(keys) < 2 + int(condense_seed):
        # TODO
        keys = ("flavor",) + keys
    if len(keys) > 2 + int(condense_seed):
        # TODO: condense excess keys into Cartesian product
        raise NotImplementedError("handle other amounts of keys")

    learning_curves = [lib.load_learning_curve(s, rootdir) for s in specs]
    # TODO not magic number
    last_k = min(learning_curves[0].shape[0], 5)
    train_rews = [lc[-last_k:].flatten() for lc in learning_curves]

    results = lib.load_test_results(multispec, rootdir)
    assert len(results) == len(specs)
    import pdb; pdb.set_trace()
    assert type(results[0][1]) == list
    assert type(results[0][1][0]) == dict
    test_rews = [np.array(list(lib.iter_key(segs, "ep_rews"))).flatten() for _, segs in results]
    assert len(test_rews) == len(specs)

    fig = plt.figure(figsize=(6,10))
    plt.style.use("seaborn-dark")

    multivals, train_table = lib.tabulate(specs, train_rews, keys)
    _, test_table = lib.tabulate(specs, test_rews, keys)

    # collapse seed axis (TODO: more numpy-ish way?)
    if "seed" in keys and condense_seed:
        iseed = keys.index("seed")
        order = range(iseed) + range(iseed + 1, len(keys)) + [iseed]
        train_table = lib.flatlast(train_table.permute(order))
        test_table = lib.flatlast(test_table.permute(order))

    sh = train_table.shape[:2]
    assert test_table.shape[:2] == sh
    train_table = train_table.reshape((sh) + (-1,))
    test_table = test_table.reshape((sh) + (-1,))

    # TODO move to lib ?
    namesub = {
        "embed_tanh": "tanh(e)",
        "alpha_sysid": "\\alpha",
    }
    multivals = [[f"{namesub.get(k, k)} = {v}" for v in mv] for k, mv in zip(keys, multivals)]
    labels = multivals + [("train", "test")]
    fig = plots.nested_boxplot(labels, train_table, test_table, aspect_ratio=1.2)
    fig.savefig(outpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="load experiments results and make boxplot")
    parser.add_argument("rootdir", type=str)
    parser.add_argument("outpath", type=str)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--condense-seed", action="store_true")
    args = parser.parse_args()

    main(args.rootdir, args.outpath, args.condense_seed, args.transpose)
