"""Diagnostic: describe .mat files (keys, shapes) and the folder layout under a directory."""
import collections
import glob
import os
import sys

import scipy.io as sio

d = sys.argv[1]
dirs = collections.Counter(os.path.dirname(os.path.relpath(f, d)) for f in glob.glob(f"{d}/**/*.mat", recursive=True))
print(dict(dirs))
for f in sorted(glob.glob(f"{d}/**/*.mat", recursive=True))[:2] + sorted(glob.glob(f"{d}/EC/*/s*.mat"))[:1]:
    try:
        m = sio.loadmat(f)
        print(os.path.relpath(f, d), {k: (getattr(v, "shape", None), getattr(v, "dtype", None)) for k, v in m.items() if not k.startswith("__")})
    except NotImplementedError:
        import h5py
        with h5py.File(f) as h:
            print(os.path.relpath(f, d), "v7.3", {k: h[k].shape for k in h.keys()})
