"""List the TDBRAIN V3.1 archive layout (password from $TDBRAIN_PW). Diagnostic only."""
import collections
import subprocess
import sys

z = sys.argv[1]
out = subprocess.run(["bsdtar", "-tf", z], capture_output=True, text=True)
names = out.stdout.splitlines()
print("entries", len(names), "stderr", out.stderr[:300])
ext = collections.Counter(n.rsplit(".", 1)[-1] for n in names if "." in n.rsplit("/", 1)[-1])
print(dict(ext))
for n in names[:25]:
    print(n)
print([n for n in names if "rest" in n.lower()][:12])
