#!/bin/bash
# Revision 2 close-out: wait for the TUAB repeat loops, sync + check them, score on Eval, then
# regenerate statistics and every report/figure.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
# wait until BOTH runs are positively reported finished (a failed status call must not end the wait)
until [ "$(./core/run.sh python core/remote.py status 2>/dev/null | grep -cE 'runs_repeat/abnormal_tuab/s1r[23] finished')" = "2" ]; do
  sleep 300
done
./core/run.sh python core/remote.py sync eeg_psych subdir=runs_repeat
for r in s1r2 s1r3; do
  n=$(wc -l < projects/eeg_psych/runs_repeat/abnormal_tuab/$r/history.jsonl)
  echo "abnormal_tuab $r rounds=$n"
done
./core/run.sh python core/final.py eeg_psych abnormal_tuab --subdir runs_repeat --runs s1r2 s1r3
./core/run.sh python core/stats.py
python3 core/report_robustness.py
python3 core/report.py
echo FINISHED
