import argparse
import itertools as it
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import libexperiment as lib
import plots

def main(rootdir, outpath, transpose=False):

    spec_path = os.path.join(rootdir, "config.json")
    with open(spec_path) as f:
        multispec = lib.Spec(json.load(f))
    rootdir = os.path.join(rootdir, "results")
    specs = [s for s in lib.multispec_product(multispec)]

    multis = multispec.multi_items()
    if "seed" in multis:
        del multis["seed"]
    keys = sorted(multis.keys())
    if transpose:
        keys = list(reversed(keys))

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

    # TODO move to lib ?
    def kv_string(k, v):
        namesub = {
            "embed_tanh": "tanh(e)",
            "alpha_sysid": "$\\alpha$",
        }
        return f"{namesub.get(k, k)} = {v}"

    labels = [[kv_string(k, v) for v in mv] for k, mv in zip(keys, multivals)]

    if len(keys) > 2:
        collapsed = list(it.product(*labels[:-1]))
        newshape = (len(collapsed), len(labels[-1]), -1)
        labels = [["\n".join(t) for t in collapsed]] + labels[-1:]
    else:
        newshape = tuple(len(mv) for mv in multivals) + (-1,)

    train_table = train_table.reshape(newshape)
    test_table = test_table.reshape(newshape)
    labels += [("train", "test")]

    if len(labels) == 2:
        print("prepending dummy dimension")
        # TODO actually handle this well
        labels = [("dummy_1", "dummy_2")] + labels
        keys = ("dummy",) + keys
        train_table = np.stack([train_table, train_table])
        test_table = np.stack([test_table, test_table])

    fig = plots.nested_boxplot(labels, train_table, test_table, aspect_ratio=1.2)
    fig.savefig(outpath)

    return

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
    args = parser.parse_args()

    main(args.rootdir, args.outpath, args.transpose)
