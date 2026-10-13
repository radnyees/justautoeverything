"""core/bundle.py -- pack core/ + one project's code (no data, no runs) for a Modal call."""
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"runs", "__pycache__", "results", "figures", "archive"}


def bundle(project):
    """project=None: core/ + every project's code."""
    buf = io.BytesIO()
    bases = [ROOT / "core"] + ([ROOT / "projects" / project] if project else [ROOT / "projects"])
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for base in bases:
            for f in base.rglob("*"):
                rel = f.relative_to(ROOT)
                if f.is_file() and not (set(rel.parts) & SKIP) and not any(p.startswith("runs_") for p in rel.parts) \
                        and f.suffix in {".py", ".json", ".md", ".txt", ".tsv", ".csv"} \
                        and f.stat().st_size < 5_000_000:
                    tf.add(f, arcname=str(rel))
    return buf.getvalue()
