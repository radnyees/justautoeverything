#!/bin/bash
# Runs the non-LLM search baseline (core/search.py) for all 12 tasks x 3 splits, resumable.
cd "$(dirname "$0")/.."
mkdir -p logs
./core/run.sh python core/search.py microbiome "crc_baxter ibd_pooled" "1 2 3" > logs/search_microbiome.log 2>&1 &
./core/run.sh python core/search.py proteomics_ad "plasma_ad csf_ad" "1 2 3" > logs/search_proteomics_ad.log 2>&1 &
./core/run.sh python core/search.py drug_response "prism_auc" "1 2 3" > logs/search_drug_response.log 2>&1 &
./core/run.sh python core/search.py fmri_connectivity "autism_abide adhd_adhd200" "1 2 3" > logs/search_fmri_connectivity.log 2>&1 &
./core/run.sh python core/search.py eeg_psych "mdd_tdbrain adhd_tdbrain sz_aseeg abnormal_tuab" "1 2 3" > logs/search_eeg_psych.log 2>&1 &
./core/run.sh python core/search.py neonatal_seizure "helsinki" "1 2 3" > logs/search_neonatal_seizure.log 2>&1 &
wait
