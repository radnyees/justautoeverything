"""
core/loop.py -- the autoresearch loop, shared by every project.

ONE editable file (a task's `solution.py`), ONE scalar metric (the harness's Dev score),
keep-if-better-else-discard, repeated until no improvement for --patience rounds.

Each round:
  1. Ask the LLM (headless Claude Code CLI) -- given core/brief.md, the project's program.md,
     the task note, the current best solution and the full history -- for a complete new
     solution plus a one-line rationale and the feature families it uses.
  2. Score it on Modal through core/harness.py (Dev = grouped K-fold CV inside the dev pool).
  3. Keep it iff it beats the best by > EPS; otherwise discard it and tell the LLM.
  4. Append the outcome to <run_dir>/history.jsonl.

Safety properties (carried over from autoepi):
  * candidates are written to candidate.py; solution.py is ALWAYS the champion, so an
    interrupted round can never be resumed from an unscored candidate;
  * runs resume from history.jsonl after any interruption;
  * a Modal account/network outage is waited out and never recorded as a failed round;
  * failure reasons, model and reasoning effort are recorded per round.

    ./core/run.sh python core/loop.py --project eeg_psych --task mdd_mumtaz --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bundle import bundle  # noqa: E402

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", str(Path.home() / ".local/bin/claude"))
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
EPS = 1e-4
NEG_INF = float("-inf")
# Modal errors that are infrastructure (account disabled, auth, network), not the candidate
INFRA_RE = re.compile(r"disabled|billing|quota|unauthori[sz]ed|authenticat|token|connection|"
                      r"unavailable|rate.?limit|ConnectionError|StreamTerminated|"
                      r"timed out connecting", re.IGNORECASE)
INFRA_RETRIES, INFRA_WAIT = 24, 300

SYSTEM = (
    "You are an autonomous ML researcher improving a single Python file for a biomedical "
    "prediction task. You optimize one metric (higher is better), measured by a fixed "
    "harness you cannot change. You propose ONE focused, well-motivated change per round.\n\n"
    "Respond in exactly this format:\n"
    "RATIONALE: <one or two sentences on the single change you are making and why>\n"
    "FAMILIES: <comma-separated feature/model families the NEW solution relies on, chosen "
    "from the project's family list>\n"
    "```python\n<the COMPLETE new contents of solution.py>\n```"
)


def project_cfg(project):
    return json.loads((ROOT / "projects" / project / "project.json").read_text())


def _llm_slot():
    """Hold one of JAE_LLM_SLOTS machine-wide slots while a claude CLI call runs (each CLI
    process needs ~300 MB; the orchestrator box is small). Returns an open locked file."""
    import fcntl
    n = int(os.environ.get("JAE_LLM_SLOTS", "3"))
    d = Path("/tmp/jae_llm_slots")
    d.mkdir(exist_ok=True)
    while True:
        for i in range(n):
            f = open(d / f"slot{i}", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except OSError:
                f.close()
        time.sleep(5)


def ask_llm(prompt, model, timeout, effort=None):
    slot = _llm_slot()
    try:
        return _ask_llm(prompt, model, timeout, effort)
    finally:
        slot.close()


def _ask_llm(prompt, model, timeout, effort=None):
    cmd = [CLAUDE_BIN, "-p", "--model", model, "--append-system-prompt", SYSTEM,
           "--disallowed-tools", "Bash", "Edit", "Write", "Read", "NotebookEdit",
           "WebFetch", "WebSearch", "Agent"]
    if effort:
        cmd += ["--effort", effort]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
                       cwd="/tmp")
    if p.returncode != 0:
        raise RuntimeError(f"claude CLI failed ({p.returncode}): {p.stderr[-600:]} {p.stdout[-600:]}")
    return p.stdout


def evaluate(project, task, size, code, seed, mode="dev", report=False, time_limit=1500):
    """Score a solution on Modal. Returns (score|-inf, log, report_json|None)."""
    import modal
    fn = modal.Function.from_name("jae", f"ev_{size}")
    for attempt in range(1, INFRA_RETRIES + 1):
        try:
            res = fn.remote(project, task, code, mode, seed, report, time_limit, bundle(project))
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if not INFRA_RE.search(msg):          # e.g. container OOM -> the candidate's fault
                return NEG_INF, f"MODAL ERROR: {msg[-1500:]}", None
            print(f"[loop] Modal unavailable ({attempt}/{INFRA_RETRIES}): {msg[:200]}", flush=True)
            if attempt == INFRA_RETRIES:
                sys.exit("[loop] Modal still unavailable; stopping without recording the round "
                         "(re-run to resume)")
            time.sleep(INFRA_WAIT)
    if res.get("score") is None:
        return NEG_INF, res.get("log", "no score")[-2000:], None
    return float(res["score"]), (res.get("log") or "")[-600:], res.get("report")


def parse_reply(text):
    rat = fam = ""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("RATIONALE:") and not rat:
            rat = s.split(":", 1)[1].strip()
        elif s.upper().startswith("FAMILIES:") and not fam:
            fam = s.split(":", 1)[1].strip()
    blocks = CODE_RE.findall(text)
    return rat, [f.strip() for f in fam.split(",") if f.strip()], (blocks[-1].strip() if blocks else None)


def build_prompt(brief, program, task, tcfg, families, best_code, best, history, last_fail):
    parts = [brief, "\n\n", program,
             f"\n\n## Active task: `{task}`\n{tcfg.get('note', '')}\n",
             f"\nMetric: {tcfg.get('metric', 'AUROC')} (higher is better), "
             f"Dev = {tcfg.get('cv_folds', 5)}-fold grouped CV inside the dev pool.\n",
             "\nFeature/model families to tag in FAMILIES: " + ", ".join(families) + "\n",
             f"\n# Current best solution.py (Dev score = {best:.6f})\n```python\n{best_code}\n```\n"]
    if history:
        parts.append("\n# History -- every round so far (most recent last). Do NOT re-propose "
                     "an idea already listed; build on what was KEPT.\n")
        for h in history:
            sc = "FAILED" if h["score"] == NEG_INF else f"{h['score']:.6f} ({h['delta']:+.6f})"
            parts.append(f"- round {h['round']}: {'KEPT' if h['kept'] else 'discarded'} | {sc} | "
                         f"[{', '.join(h.get('families', []))}] {h['rationale']}\n")
    if last_fail:
        parts.append("\n# Your previous attempt FAILED. Fix it or try a different idea.\n"
                     f"Error:\n```\n{last_fail['log'][-1800:]}\n```\n")
    parts.append("\nPropose the next single improvement: RATIONALE, FAMILIES, then the complete "
                 "new solution.py in one ```python block.")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=1, help="split seed (1, 2, 3 ...)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--time-limit", type=int, default=1500, help="seconds per candidate")
    ap.add_argument("--llm-timeout", type=int, default=1200)
    ap.add_argument("--llm-retries", type=int, default=6)
    ap.add_argument("--strategy", default="loop", choices=["loop", "best_of_n"],
                    help="loop = iterative keep-if-better with full history (the method); best_of_n = "
                         "baseline: every round sees only the brief + the SEED solution (no history, no "
                         "feedback); the best Dev candidate is kept. Round 1 of best_of_n = one-shot baseline.")
    ap.add_argument("--runs-subdir", default="runs", help="runs | runs_bestof20 | runs_repeat ...")
    ap.add_argument("--run-name", default=None, help="run folder name (default s<seed>)")
    args = ap.parse_args()

    cfg = project_cfg(args.project)
    tcfg = cfg["tasks"][args.task]
    size = tcfg.get("size", "m")
    pdir = ROOT / "projects" / args.project
    runs_root = Path(os.environ.get("JAE_RUNS_ROOT", ROOT / "projects"))   # /runs on Modal
    run_dir = runs_root / args.project / args.runs_subdir / args.task / (args.run_name or f"s{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    solution, candidate = run_dir / "solution.py", run_dir / "candidate.py"
    history_path = run_dir / "history.jsonl"
    if not solution.exists():
        seed_file = pdir / "seeds" / f"{args.task}.py"
        seed_file = seed_file if seed_file.exists() else pdir / "seeds" / "default.py"
        solution.write_text(seed_file.read_text())
        print(f"[loop] seeded {solution} from {seed_file.name}")

    brief = (ROOT / "core" / "brief.md").read_text()
    program = (pdir / "program.md").read_text()
    families = cfg.get("families", [])

    best, log, _ = evaluate(args.project, args.task, size, solution.read_text(), args.seed,
                            time_limit=args.time_limit)
    if best == NEG_INF:
        sys.exit(f"[loop] the current solution does not run:\n{log}")
    best_code = solution.read_text()
    seed_code, seed_score = best_code, best                 # what best_of_n always shows the LLM
    if args.strategy == "best_of_n" and (run_dir / "baseline.json").exists():
        seed_code = (pdir / "seeds" / f"{args.task}.py") if (pdir / "seeds" / f"{args.task}.py").exists() \
            else (pdir / "seeds" / "default.py")
        seed_code = seed_code.read_text()
        seed_score = json.loads((run_dir / "baseline.json").read_text())["dev"]
    (run_dir / "score.txt").write_text(f"{best:.6f}\n")
    if not (run_dir / "baseline.json").exists():          # the seed's Dev score, recorded once
        (run_dir / "baseline.json").write_text(json.dumps({"dev": best, "model": args.model,
                                                            "effort": args.effort}) + "\n")
    print(f"[loop] {args.project}/{args.task} s{args.seed}: baseline Dev = {best:.6f}", flush=True)

    history = [json.loads(l) for l in history_path.read_text().splitlines() if l.strip()] \
        if history_path.exists() else []
    start = max((h["round"] for h in history), default=0) + 1
    stale = 0
    for h in reversed(history):
        if h["kept"]:
            break
        stale += 1
    if history:
        print(f"[loop] RESUMING at round {start} (stale={stale}, best={best:.6f})", flush=True)
    last_fail = None

    for rnd in range(start, args.rounds + 1):
        if args.strategy == "loop" and stale >= args.patience:
            print(f"[loop] no improvement in {args.patience} rounds -- converged.")
            break
        if args.strategy == "loop":
            prompt = build_prompt(brief, program, args.task, tcfg, families, best_code, best,
                                  history, last_fail)
        else:                                               # independent sample: no history/feedback
            prompt = build_prompt(brief, program, args.task, tcfg, families, seed_code, seed_score,
                                  [], None)
        reply = None
        for attempt in range(1, args.llm_retries + 1):
            try:
                reply = ask_llm(prompt, args.model, args.llm_timeout, args.effort)
                break
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"[loop] round {rnd}: LLM call failed ({attempt}): {str(e)[:200]}", flush=True)
                time.sleep(60 * attempt)                     # back off (rate limits, network)
        if reply is None:
            print(f"[loop] round {rnd}: LLM unreachable; skipping (not counted).", flush=True)
            continue
        rationale, fams, code = parse_reply(reply)
        if not code:
            print(f"[loop] round {rnd}: no code block; skipping.", flush=True)
            continue

        candidate.write_text(code)
        (run_dir / "candidates").mkdir(exist_ok=True)
        (run_dir / "candidates" / f"round_{rnd:02d}.py").write_text(code)   # every candidate kept
        t0 = time.time()
        score, log, _ = evaluate(args.project, args.task, size, code, args.seed,
                                 time_limit=args.time_limit)
        secs = round(time.time() - t0)
        delta = 0.0 if score == NEG_INF else score - best
        kept = score > best + EPS
        if kept:
            best, best_code, stale, last_fail = score, code, 0, None
            solution.write_text(code)
            (run_dir / "score.txt").write_text(f"{best:.6f}\n")
        else:
            stale += 1
            last_fail = {"log": log} if score == NEG_INF else None
        shown = "FAILED" if score == NEG_INF else f"{score:.6f}"
        print(f"[loop] round {rnd}: {'KEPT' if kept else 'discarded'} {shown} "
              f"(best {best:.6f}, stale {stale}) | {rationale}", flush=True)
        rec = {"round": rnd, "rationale": rationale, "families": fams, "score": score,
               "delta": delta, "kept": kept, "seconds": secs, "model": args.model,
               "effort": args.effort, "strategy": args.strategy,
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if score == NEG_INF:
            rec["fail_log"] = log[-1500:]
        history.append(rec)
        with open(history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"[loop] done. best Dev = {best:.6f} ({solution})", flush=True)


if __name__ == "__main__":
    main()
