import argparse
import json
import os
import sys

import libexperiment as lib

def main():
    parser = argparse.ArgumentParser(description="train rl-sysid policies based on json config.")
    parser.add_argument("--procs", type=int, default=0)
    parser.add_argument("rootdir", type=str)
    args = parser.parse_args()

    specpath = os.path.join(args.rootdir, "config.json")
    with open(specpath) as f:
        spec = lib.Spec(json.load(f))

    rootdir = os.path.join(args.rootdir, "results")
    if args.procs == 0:
        args.procs = os.cpu_count() // 2 - 1
    lib.train_multispec(spec, rootdir, args.procs)


if __name__ == "__main__":
    main()
