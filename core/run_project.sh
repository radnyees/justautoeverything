#!/usr/bin/env bash
# core/run_project.sh -- run a project's loops in parallel (one per task x split seed).
#   core/run_project.sh microbiome "crc_baxter ibd_gevers" "1 2 3" [rounds] [patience]
# Every loop is resumable: re-running continues from history.jsonl.
cd "$(dirname "$0")/.."
P=$1; TASKS=$2; SEEDS=${3:-"1 2 3"}; R=${4:-20}; PAT=${5:-10}
for T in $TASKS; do for S in $SEEDS; do
  mkdir -p projects/$P/runs/$T
  ( ./core/run.sh python -u core/loop.py --project $P --task $T --seed $S --rounds $R --patience $PAT \
      >> projects/$P/runs/$T/s$S.log 2>&1; echo "[$P/$T/s$S] exit $?" ) &
  sleep 15
done; done
wait
