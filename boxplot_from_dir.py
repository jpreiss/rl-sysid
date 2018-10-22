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
    if len(keys) > 2 + int(condense_seed):
        # TODO: condense excess keys into Cartesian product
        raise NotImplementedError("handle other amounts of keys")

    learning_curves = [lib.load_learning_curve(s, rootdir) for s in specs]
    # TODO not magic number
    last_k = min(learning_curves[0].shape[0], 5)
    train_rews = [lc[-last_k:].flatten() for lc in learning_curves]

    results = lib.load_test_results(multispec, rootdir)
    assert len(results) == len(specs)
    assert type(results[0][1]) == list
    assert type(results[0][1][0]) == dict
    test_rews = [np.array(list(lib.iter_key(segs, "ep_rews"))).flatten() for _, segs in results]
    assert len(test_rews) == len(specs)

    fig = plt.figure(figsize=(6,10))
    plt.style.use("seaborn-dark")


    multivals, train_table = lib.tabulate(specs, train_rews, keys)
    _, test_table = lib.tabulate(specs, test_rews, keys)
    # TODO why does tabulate add a singleton dimension for (flavor, seed)

    # collapse seed axis (TODO: more numpy-ish way?)
    if "seed" in keys and condense_seed:
        raise NotImplementedError
        iseed = keys.index("seed")
        order = range(iseed) + range(iseed + 1, len(keys)) + [iseed]
        train_table = lib.flatlast(train_table.permute(order))
        test_table = lib.flatlast(test_table.permute(order))

    sh = train_table.shape[:len(keys)]
    assert test_table.shape[:len(keys)] == sh
    train_table = train_table.reshape((sh) + (-1,))
    test_table = test_table.reshape((sh) + (-1,))

    # TODO move to lib ?
    namesub = {
        "embed_tanh": "tanh(e)",
        "alpha_sysid": "\\alpha",
    }
    multivals = [[f"{namesub.get(k, k)} = {v}" for v in mv] for k, mv in zip(keys, multivals)]
    labels = multivals + [("train", "test")]
    if len(multivals) == 1:
        print("prepending dummy dimension")
        # TODO actually handle this well
        labels = [("foo", "bar")] + labels
        keys = ("dummy",) + keys
        train_table = np.stack([train_table, train_table])
        test_table = np.stack([test_table, test_table])
    fig = plots.nested_boxplot(labels, train_table, test_table, aspect_ratio=1.2)
    fig.savefig(outpath)

    texpath = os.path.splitext(outpath)[0] + ".csv"
    columns = {
        "median" : lambda r: np.median(r.flat),
        "mean"   : lambda r: np.mean(r.flat),
        "min"    : lambda r: np.min(r.flat),
        "max"    : lambda r: np.max(r.flat),
        "std"    : lambda r: np.std(r.flat),
    }
    with open(texpath, "w") as f:
        cols = it.product(*labels)
        labxp = labels[-1:] + labels[:-1]
        f.write("train_test, " + ", ".join(keys + tuple(columns.keys())) + "\n")

        def rec(labels, table):
            if labels == []:
                cells = [func(table) for func in columns.values()]
                cellstrs = [f"{x:.1f}" for x in cells]
                yield ", ".join(cellstrs) + "\n"
            else:
                for lab, tab in zip(labels[0], table):
                    for row in rec(labels[1:], tab):
                        yield f"{lab}, " + row

        for row in rec(labxp, [train_table, test_table]):
            f.write(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="load experiments results and make boxplot")
    parser.add_argument("rootdir", type=str)
    parser.add_argument("outpath", type=str)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--condense-seed", action="store_true")
    args = parser.parse_args()

    main(args.rootdir, args.outpath, args.condense_seed, args.transpose)
