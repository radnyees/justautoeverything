"""
core/modal_app.py -- Modal backend for every justautoeverything project.

* ev_{s,m,l}: run ONE candidate solution through core/harness.py in a container of
  the project's size class and return its score (+ optional prediction report).
* pr_{m,l}: run a project's one-time data-building script (download + preprocess into the
  shared `jae-data` volume) with a long timeout.

Deploy once (and after editing harness/prepare/data code):  ./core/run.sh modal deploy core/modal_app.py
"""
import os

import modal

app = modal.App("jae")
data = modal.Volume.from_name("jae-data", create_if_missing=True)
DATA = "/data"
REPO = "/root/jae"

base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "unzip", "rsync", "openssh-client", "unar", "libarchive-tools")
    .pip_install(
        "numpy<2.3", "scipy", "pandas", "pyarrow", "scikit-learn", "statsmodels",
        "lightgbm", "xgboost", "catboost", "networkx", "PyWavelets", "antropy",
        "mne", "mne-connectivity", "pyriemann", "nilearn", "h5py", "pyedflib",
        "requests", "boto3", "tqdm", "joblib", "openpyxl",
    )
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .env({"JAE_DATA": DATA, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "0"})
)
image = (
    base.add_local_dir(
        os.path.join(os.path.dirname(__file__), ".."), REPO,
        ignore=["**/runs/**", "**/runs_*/**", "results/**", ".git/**", "**/__pycache__/**",
                "**/*.png", "**/*.pdf", "**/candidate.py", "catboost_info/**"],
    )
)


def _repo(bundle):
    """Unpack the caller's code bundle (core/ + the project's code) so every call runs exactly
    the caller's current code -- warm containers from an older deployment can't serve stale code."""
    import io
    import tarfile
    import tempfile
    if not bundle:
        return REPO
    root = tempfile.mkdtemp(prefix="jae_")
    tarfile.open(fileobj=io.BytesIO(bundle)).extractall(root)
    return root


def _evaluate(project, task, code, mode="dev", seed=1, report=False, time_limit=1500, bundle=None):
    import re
    import subprocess
    import sys
    import tempfile

    data.reload()
    repo = _repo(bundle)
    work = tempfile.mkdtemp()
    sol = os.path.join(work, "solution.py")
    with open(sol, "w") as f:
        f.write(code)
    rep = os.path.join(work, "report.json")
    env = {**os.environ, "JAE_PROJECT": project, "JAE_TASK": task, "JAE_SEED": str(seed),
           "JAE_MODE": mode, "PYTHONPATH": repo}
    if report:
        env.update(JAE_REPORT="1", JAE_REPORT_PATH=rep)
    try:
        p = subprocess.run([sys.executable, f"{repo}/core/harness.py", sol], cwd=work, env=env,
                           capture_output=True, text=True, timeout=time_limit)
    except subprocess.TimeoutExpired:
        return {"score": None, "log": f"TIMEOUT after {time_limit}s", "report": None}
    out = p.stdout + p.stderr
    if p.returncode != 0:
        return {"score": None, "log": "CRASH:\n" + out[-2500:], "report": None}
    m = re.search(r"^SCORE:\s*([-+0-9.eE]+|nan)\s*$", p.stdout, re.M)
    if not m or m.group(1) == "nan":
        return {"score": None, "log": "NO valid SCORE line:\n" + out[-2500:], "report": None}
    rj = open(rep).read() if report and os.path.exists(rep) else None
    return {"score": float(m.group(1)), "log": out[-1500:], "report": rj}


KW = dict(image=image, volumes={DATA: data}, timeout=3900, scaledown_window=120)


@app.function(cpu=2.0, memory=8192, **KW)
def ev_s(*a, **k):
    return _evaluate(*a, **k)


@app.function(cpu=8.0, memory=32768, **KW)
def ev_m(*a, **k):
    return _evaluate(*a, **k)


@app.function(cpu=16.0, memory=65536, **KW)
def ev_l(*a, **k):
    return _evaluate(*a, **k)


def _prep(script, args=(), bundle=None):
    import subprocess
    import sys
    data.reload()
    repo = _repo(bundle)
    env = {**os.environ, "PYTHONPATH": repo}
    p = subprocess.run([sys.executable, f"{repo}/{script}", *args], cwd=repo, env=env,
                       capture_output=True, text=True)
    data.commit()
    return {"returncode": p.returncode, "log": (p.stdout + p.stderr)[-6000:]}


PREP = dict(image=image, volumes={DATA: data}, timeout=6 * 3600)


@app.function(cpu=8.0, memory=32768, **PREP)
def pr_m(script, args=(), bundle=None):
    return _prep(script, args, bundle)


@app.function(cpu=16.0, memory=65536, **PREP)
def pr_l(script, args=(), bundle=None):
    return _prep(script, args, bundle)


# ---------------------------------------------------------------- remote loop orchestration
runs = modal.Volume.from_name("jae-runs", create_if_missing=True)
orch_image = base.run_commands(
    "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
    "apt-get install -y nodejs",
    "npm i -g @anthropic-ai/claude-code@2.1.296",
).pip_install("modal")


@app.function(image=orch_image, volumes={"/runs": runs}, secrets=[modal.Secret.from_name("jae-claude")],
              cpu=0.5, memory=2048, timeout=24 * 3600)
def orchestrate(project, task, seed, rounds=20, patience=10, effort="high", model="claude-opus-5-5",
                bundle=None, extra=()):
    """Run one autoresearch loop (core/loop.py) inside Modal; history lives on the jae-runs volume
    (/runs/<project>/runs/<task>/s<seed>/), committed every minute so nothing is lost."""
    import shutil
    import subprocess
    import sys
    import threading

    runs.reload()
    repo = _repo(bundle)
    sub = dict(zip(extra[::2], extra[1::2])).get("--runs-subdir", "runs")
    name = dict(zip(extra[::2], extra[1::2])).get("--run-name", f"s{seed}")
    log_dir = f"/runs/{project}/{sub}/{task}"
    os.makedirs(log_dir, exist_ok=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(60):
            try:
                runs.commit()
            except Exception:
                pass
    threading.Thread(target=committer, daemon=True).start()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [repo, os.environ.get("PYTHONPATH")])),
           "JAE_RUNS_ROOT": "/runs", "JAE_LLM_SLOTS": "1",
           "CLAUDE_BIN": shutil.which("claude") or "claude", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen([sys.executable, "-u", f"{repo}/core/loop.py", "--project", project, "--task", task,
                          "--seed", str(seed), "--rounds", str(rounds), "--patience", str(patience),
                          "--effort", effort, "--model", model, *extra],
                         cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:                       # append + close per line so commits always see it
        with open(f"{log_dir}/{name}.log", "a") as log:
            log.write(line)
    p.wait()
    stop.set()
    runs.commit()
    return p.returncode
