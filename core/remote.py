"""
core/remote.py -- run autoresearch loops ON MODAL (orchestrator containers) and sync results back.

    ./core/run.sh python core/remote.py launch eeg_psych "mdd_tdbrain adhd_tdbrain" "1 2 3"
    ./core/run.sh python core/remote.py sync eeg_psych        # copy /runs/<project>/runs -> projects/<project>/runs
    ./core/run.sh python core/remote.py status
Launched calls are recorded in results/remote_calls.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bundle import bundle  # noqa: E402

CALLS = ROOT / "results" / "remote_calls.json"


def launch(project, tasks, seeds, rounds=20, patience=10, effort="high", strategy="loop",
           subdir="runs", repeats=""):
    """seeds: split seeds; repeats (optional, e.g. "2 3"): extra independent loop runs on each
    seed, stored as s<seed>r<k> in <subdir>."""
    fn = modal.Function.from_name("jae", "orchestrate")
    calls = json.loads(CALLS.read_text()) if CALLS.exists() else {}
    b = bundle(project)
    for t in tasks.split():
        for s in seeds.split():
            for r in (repeats.split() or [None]):
                name = f"s{s}" + (f"r{r}" if r else "")
                extra = ["--strategy", strategy, "--runs-subdir", subdir, "--run-name", name]
                c = fn.spawn(project, t, int(s), int(rounds), int(patience), effort, "claude-opus-5-5", b,
                             extra)
                calls[f"{project}/{subdir}/{t}/{name}"] = c.object_id
                print("launched", project, subdir, t, name, c.object_id, flush=True)
    CALLS.parent.mkdir(exist_ok=True)
    CALLS.write_text(json.dumps(calls, indent=1))


def sync(project, subdir="runs"):
    """MERGE the remote run records into projects/<project>/runs: files on the volume are copied
    over local ones, but local-only files (final_scores.json, final/ predictions written by
    core/final.py) are never deleted."""
    import shutil
    import tempfile
    dst = ROOT / "projects" / project
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["modal", "volume", "get", "--force", "jae-runs", f"/{project}/{subdir}", tmp], check=True)
        src = Path(tmp) / subdir
        if src.exists():
            shutil.copytree(src, dst / subdir, dirs_exist_ok=True)
    print("synced", project)


def status():
    calls = json.loads(CALLS.read_text()) if CALLS.exists() else {}
    for k, cid in calls.items():
        fc = modal.FunctionCall.from_id(cid)
        try:
            rc = fc.get(timeout=0)
            print(k, "finished rc", rc)
        except TimeoutError:
            print(k, "running")
        except Exception as e:
            print(k, "error", type(e).__name__, str(e)[:120])


if __name__ == "__main__":
    cmd = sys.argv[1]
    kw = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    pos = [a for a in sys.argv[2:] if "=" not in a]
    {"launch": lambda: launch(*pos, **kw), "sync": lambda: sync(*pos, **kw), "status": status}[cmd]()
