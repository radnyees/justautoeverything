#!/bin/bash
# Seed-derived ("independent") Optuna baseline (docs/preregistration_phase3.md), all 12 tasks x 3 splits.
cd "$(dirname "$0")/.."
mkdir -p logs
S="--space-dir baselines_indep --subdir runs_optuna_indep"
./core/run.sh python core/search.py microbiome "crc_baxter ibd_pooled" "1 2 3" $S > logs/indep_microbiome.log 2>&1 &
./core/run.sh python core/search.py proteomics_ad "plasma_ad csf_ad" "1 2 3" $S > logs/indep_proteomics_ad.log 2>&1 &
./core/run.sh python core/search.py drug_response "prism_auc" "1 2 3" $S > logs/indep_drug_response.log 2>&1 &
./core/run.sh python core/search.py fmri_connectivity "autism_abide adhd_adhd200" "1 2 3" $S > logs/indep_fmri_connectivity.log 2>&1 &
./core/run.sh python core/search.py eeg_psych "mdd_tdbrain adhd_tdbrain sz_aseeg abnormal_tuab" "1 2 3" $S > logs/indep_eeg_psych.log 2>&1 &
./core/run.sh python core/search.py neonatal_seizure "helsinki" "1 2 3" $S > logs/indep_neonatal_seizure.log 2>&1 &
wait
