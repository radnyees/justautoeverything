#!/bin/bash
# Phase 3 close-out: Eval scoring of the seed-derived Optuna arm (once), then statistics and reports.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
# batches of <= 4 concurrent scorers (2 GB sandbox)
./core/run_final_baselines.sh optuna_indep "microbiome:crc_baxter microbiome:ibd_pooled proteomics_ad:plasma_ad proteomics_ad:csf_ad" > logs/final_indep_1.log 2>&1
./core/run_final_baselines.sh optuna_indep "drug_response:prism_auc fmri_connectivity:autism_abide fmri_connectivity:adhd_adhd200 eeg_psych:mdd_tdbrain" > logs/final_indep_2.log 2>&1
./core/run_final_baselines.sh optuna_indep "eeg_psych:adhd_tdbrain eeg_psych:sz_aseeg eeg_psych:abnormal_tuab neonatal_seizure:helsinki" > logs/final_indep_3.log 2>&1
n=$(ls projects/*/runs_optuna_indep/*/s*/final_scores.json | wc -l); echo "scored: $n / 36"; [ "$n" = 36 ]
./core/run.sh python core/stats.py
python3 core/report_robustness.py
echo FINISHED
