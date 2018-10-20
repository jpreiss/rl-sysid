import argparse
import json
import os
import sys

import libexperiment as lib

def main():
    parser = argparse.ArgumentParser(description="test rl-sysid policies based on json config.")
    parser.add_argument("--procs", type=int, default=6)
    parser.add_argument("--sysid-samples", type=int, default=4)
    parser.add_argument("spec", type=str)
    args = parser.parse_args()

    rootdir, name = os.path.split(args.spec)
    assert name.endswith(".json")
    with open(args.spec) as f:
        spec = lib.Spec(json.load(f))

    rootdir = os.path.join(rootdir, "results")
    lib.test_multispec(spec, rootdir, args.sysid_samples, args.procs)


if __name__ == "__main__":
    main()
