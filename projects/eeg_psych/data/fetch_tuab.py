"""
projects/eeg_psych/data/fetch_tuab.py -- rsync the TUH Abnormal EEG Corpus (TUAB, Obeid & Picone 2016;
NEDC credentials) into /data/eeg_psych/raw/tuab/. The private SSH key registered with NEDC is passed
base64-encoded as argv[2] and written only inside the container.
    python projects/eeg_psych/data/fetch_tuab.py /data/eeg_psych/raw/tuab <base64-private-key>
"""
import base64
import os
import subprocess
import sys

out, key_b64 = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
key = os.path.expanduser("~/.ssh/nedc")
os.makedirs(os.path.dirname(key), exist_ok=True)
open(key, "wb").write(base64.b64decode(key_b64))
os.chmod(key, 0o600)
ssh = f"ssh -i {key} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
r = subprocess.run(["rsync", "-a", "--partial", "--stats", "-e", ssh,
                    "nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/", out + "/"],
                   capture_output=True, text=True)
os.remove(key)
print("rsync rc", r.returncode)
print(r.stdout[-2500:], r.stderr[-1500:])
