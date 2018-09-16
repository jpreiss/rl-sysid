import argparse
import json
import os
import sys

import libexperiment as lib

parser = argparse.ArgumentParser(description="train rl-sysid policies based on json config.")
parser.add_argument("--procs", type=int, default=6)
parser.add_argument("spec", type=str)
args = parser.parse_args()

rootdir, name = os.path.split(args.spec)
assert name.endswith(".json")
with open(args.spec) as f:
    spec = lib.Spec(json.load(f))

rootdir = lib.check_spec_dir(spec, rootdir)
lib.train_multispec(spec, rootdir, args.procs)
