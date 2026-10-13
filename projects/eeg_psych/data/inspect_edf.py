"""Diagnostic: print channel names / fs / duration of some raw EDF files. argv: glob patterns."""
import glob
import sys

import mne

for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat))[:3]:
        r = mne.io.read_raw(f, preload=False, verbose="ERROR")
        print(f.rsplit("/", 1)[-1], r.info["sfreq"], round(r.times[-1]), r.ch_names, flush=True)
