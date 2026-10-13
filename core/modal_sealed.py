"""
core/modal_sealed.py -- network-isolated scoring functions (separate app "jae-sealed").

Same image, data volume and harness as core/modal_app.py, but containers have NO network access
(block_network=True). Used to re-score frozen solutions in the leakage audit (core/anon_check.py);
a solution that needed the network would fail here. Deployed separately so in-flight loops on the
"jae" app are not disturbed:  ./core/run.sh modal deploy core/modal_sealed.py
"""
import modal

from modal_app import DATA, _evaluate, data, image

app = modal.App("jae-sealed")
sealed_image = image.add_local_python_source("modal_app")     # so containers can import _evaluate
KW = dict(image=sealed_image, volumes={DATA: data}, timeout=3900, scaledown_window=60, block_network=True)


@app.function(cpu=2.0, memory=8192, **KW)
def ev_s(*a, **k):
    return _evaluate(*a, **k)


@app.function(cpu=8.0, memory=32768, **KW)
def ev_m(*a, **k):
    return _evaluate(*a, **k)


@app.function(cpu=16.0, memory=65536, **KW)
def ev_l(*a, **k):
    return _evaluate(*a, **k)
