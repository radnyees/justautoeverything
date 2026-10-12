#!/usr/bin/env bash
# core/finish_when_done.sh -- wait until every launched remote loop has finished, then sync the
# given projects and run the one-shot final scoring for the given tasks.
#   core/finish_when_done.sh "fmri_connectivity:autism_abide eeg_psych:abnormal_tuab neonatal_seizure:helsinki"
cd "$(dirname "$0")/.."
while ./core/run.sh python core/remote.py status 2>/dev/null | grep -q " running"; do sleep 300; done
for pt in $1; do
  p=${pt%%:*}; t=${pt#*:}
  ./core/run.sh python core/remote.py sync "$p" >/dev/null 2>&1
  ./core/run.sh python core/final.py "$p" "$t" 2>&1 | grep "$p/\|rror"
done
python3 core/report.py > /dev/null 2>&1 && echo "report updated"
