import argparse
import json
import os
import sys

import libexperiment as lib

parser = argparse.ArgumentParser()
parser.add_argument("spec_path")
parser.add_argument("--suffix", type=str)
args = parser.parse_args()

rootdir, name = os.path.split(args.spec_path)
name, ext = os.path.splitext(name)
assert ext == ".json"
with open(args.spec_path) as f:
    spec = lib.Spec(json.load(f))

specs = lib.multispec_product(spec)
for spec in specs:
    joinargs = (args.suffix,) if args.suffix is not None else ()
    specdir = os.path.join(rootdir, spec["directory"], *joinargs)
    print(specdir)
