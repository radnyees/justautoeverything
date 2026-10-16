#!/bin/bash
# Eval scoring (once) of best-of-20 champions, one-shot candidates, Optuna champions and repeats.
#   core/run_final_baselines.sh bestof20 "microbiome:crc_baxter ..."   (arm: bestof20 | optuna | optuna_indep | repeat)
cd "$(dirname "$0")/.."
arm=$1; shift
for pt in $1; do
  p=${pt%%:*}; t=${pt#*:}
  case $arm in
    bestof20) ./core/run.sh python core/final.py $p $t --subdir runs_bestof20 &
              ./core/run.sh python core/final.py $p $t --subdir runs_bestof20 --solution candidates/round_01.py --tag oneshot & ;;
    optuna)   ./core/run.sh python core/final.py $p $t --subdir runs_optuna & ;;
    optuna_indep) ./core/run.sh python core/final.py $p $t --subdir runs_optuna_indep & ;;
    repeat)   ./core/run.sh python core/final.py $p $t --subdir runs_repeat --runs s1r2 s1r3 & ;;
  esac
done
wait
