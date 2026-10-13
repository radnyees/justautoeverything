"""core/bundle.py -- pack core/ + one project's code (no data, no runs) for a Modal call."""
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"runs", "__pycache__", "results", "figures"}


def bundle(project):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for base in [ROOT / "core", ROOT / "projects" / project]:
            for f in base.rglob("*"):
                rel = f.relative_to(ROOT)
                if f.is_file() and not (set(rel.parts) & SKIP) and f.suffix in {".py", ".json", ".md", ".txt", ".tsv", ".csv"} \
                        and f.stat().st_size < 5_000_000:
                    tf.add(f, arcname=str(rel))
    return buf.getvalue()
