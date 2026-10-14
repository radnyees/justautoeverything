#!/bin/bash
# Eval scoring and transduction audit for pooled IBD v3 (revision 2; leakage check run separately). Order matters: the main loops' final.py
# writes runs/ibd_pooled/s<k>/final/seed.json, which every baseline/repeat comparison reuses.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
./core/run.sh python core/final.py microbiome ibd_pooled > logs/final_ibd_v3_runs.log 2>&1
./core/run_final_baselines.sh bestof20 "microbiome:ibd_pooled" > logs/final_ibd_v3_bestof20.log 2>&1 &
./core/run_final_baselines.sh optuna "microbiome:ibd_pooled" > logs/final_ibd_v3_optuna.log 2>&1 &
./core/run_final_baselines.sh repeat "microbiome:ibd_pooled" > logs/final_ibd_v3_repeat.log 2>&1 &
./core/run.sh python core/audit.py microbiome "ibd_pooled" "1 2 3" > logs/audit_ibd_v3.log 2>&1 &
wait
