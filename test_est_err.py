import argparse
import itertools as it
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import libexperiment as lib
import plots

def main(rootdir):

    spec_path = os.path.join(rootdir, "config.json")
    with open(spec_path) as f:
        multispec = lib.Spec(json.load(f))
    rootdir = os.path.join(rootdir, "results")

    results = lib.load_test_results(multispec, rootdir)
    for spec, segs in results:
        print(spec["directory"])
        true = np.vstack(lib.iter_key(segs, "est_true"))
        est = np.vstack(lib.iter_key(segs, "est"))
        true = true.reshape((-1, true.shape[-1]))
        est = est.reshape((-1, est.shape[-1]))
        err = true - est
        msqerr = np.mean(err ** 2, axis=-1)
        rmse = np.sqrt(msqerr)
        worst = np.argmax(msqerr)
        plt.hist(rmse)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="load experiments results and make boxplot")
    parser.add_argument("rootdir", type=str)
    args = parser.parse_args()

    main(args.rootdir)
