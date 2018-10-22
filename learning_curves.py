import argparse
import itertools as it
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import libexperiment as lib
import plots

def main(rootdir, outpath, style, collapse_seed=True, csv_cols=None):

    spec_path = os.path.join(rootdir, "config.json")
    with open(spec_path) as f:
        multispec = lib.Spec(json.load(f))
    rootdir = os.path.join(rootdir, "results")
    specs = [s for s in lib.multispec_product(multispec)]

    curves = [lib.load_learning_curve(s, rootdir, keys=csv_cols) for s in specs]
    iters, N = curves[0].shape

    multis = multispec.multi_items()
    if "seed" in multis and collapse_seed:
        del multis["seed"]
    keys = multis.keys()

    def labels(specs, keys):
        return ["$" + s.label(keys, joiner="\n") + "$" for s in specs]

    if collapse_seed:
        def reducer(curves):
            return np.stack(curves).mean(axis=-1).T
        groups = lib.group_reduce(specs, curves, reducer) # -> List[Tuple[Spec, Any]]:
        cores, rews = zip(*groups)
        fig = plots.learning_curves(labels(cores, keys), rews, style=style)
    else:
        fig = plots.learning_curves(labels(specs, keys), curves, style=style)

    if csv_cols:
        if len(csv_cols) == 1:
            plt.ylabel(csv_cols[0])
        else:
            plt.ylabel(f"mean({', '.join(csv_cols)})")
    fig.savefig(outpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="load training log and plot learning curves")
    parser.add_argument("rootdir", type=str)
    parser.add_argument("outpath", type=str)
    parser.add_argument("--style", type=str, default="each")
    parser.add_argument("--cols", type=str, nargs="*")
    args = parser.parse_args()

    main(args.rootdir, args.outpath, args.style, collapse_seed=True, csv_cols=args.cols)
