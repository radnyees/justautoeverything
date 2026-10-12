"""
core/modal_app.py -- Modal backend for every justautoeverything project.

* evaluate_{s,m,l}: run ONE candidate solution through core/harness.py in a container of
  the project's size class and return its score (+ optional prediction report).
* prep_{m,l}: run a project's one-time data-building script (download + preprocess into the
  shared `jae-data` volume) with a long timeout.

Deploy once (and after editing harness/prepare/data code):  ./core/run.sh modal deploy core/modal_app.py
"""
import os

import modal

app = modal.App("jae")
data = modal.Volume.from_name("jae-data", create_if_missing=True)
DATA = "/data"
REPO = "/root/jae"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "unzip", "rsync", "openssh-client")
    .pip_install(
        "numpy<2.3", "scipy", "pandas", "pyarrow", "scikit-learn", "statsmodels",
        "lightgbm", "xgboost", "catboost", "networkx", "PyWavelets", "antropy",
        "mne", "mne-connectivity", "pyriemann", "nilearn", "h5py", "pyedflib",
        "requests", "boto3", "tqdm", "joblib",
    )
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .env({"JAE_DATA": DATA, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "0"})
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), ".."), REPO,
        ignore=["**/runs/**", "**/runs_*/**", "results/**", ".git/**", "**/__pycache__/**",
                "**/*.png", "**/*.pdf", "**/candidate.py", "catboost_info/**"],
    )
)


def _evaluate(project, task, code, mode="dev", seed=1, report=False, time_limit=1500):
    import re
    import subprocess
    import sys
    import tempfile

    data.reload()
    work = tempfile.mkdtemp()
    sol = os.path.join(work, "solution.py")
    with open(sol, "w") as f:
        f.write(code)
    rep = os.path.join(work, "report.json")
    env = {**os.environ, "JAE_PROJECT": project, "JAE_TASK": task, "JAE_SEED": str(seed),
           "JAE_MODE": mode, "PYTHONPATH": REPO}
    if report:
        env.update(JAE_REPORT="1", JAE_REPORT_PATH=rep)
    try:
        p = subprocess.run([sys.executable, f"{REPO}/core/harness.py", sol], cwd=work, env=env,
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


KW = dict(image=image, volumes={DATA: data}, timeout=3900)


@app.function(cpu=2.0, memory=8192, **KW)
def evaluate_s(*a, **k):
    return _evaluate(*a, **k)


@app.function(cpu=8.0, memory=32768, **KW)
def evaluate_m(*a, **k):
    return _evaluate(*a, **k)


@app.function(cpu=16.0, memory=65536, **KW)
def evaluate_l(*a, **k):
    return _evaluate(*a, **k)


def _prep(script, args=()):
    import subprocess
    import sys
    data.reload()
    env = {**os.environ, "PYTHONPATH": REPO}
    p = subprocess.run([sys.executable, f"{REPO}/{script}", *args], cwd=REPO, env=env,
                       capture_output=True, text=True)
    data.commit()
    return {"returncode": p.returncode, "log": (p.stdout + p.stderr)[-6000:]}


PREP = dict(image=image, volumes={DATA: data}, timeout=6 * 3600)


@app.function(cpu=8.0, memory=32768, **PREP)
def prep_m(script, args=()):
    return _prep(script, args)


@app.function(cpu=16.0, memory=65536, **PREP)
def prep_l(script, args=()):
    return _prep(script, args)
