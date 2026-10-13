#!/bin/bash
# Transduction audit (core/audit.py) for every task x split (seed + champion; chunked and ref-batch).
cd "$(dirname "$0")/.."
mkdir -p logs
./core/run.sh python core/audit.py microbiome "crc_baxter ibd_pooled" "1 2 3" > logs/audit_microbiome.log 2>&1 &
./core/run.sh python core/audit.py proteomics_ad "plasma_ad csf_ad" "1 2 3" > logs/audit_proteomics_ad.log 2>&1 &
./core/run.sh python core/audit.py drug_response "prism_auc" "1 2 3" > logs/audit_drug_response.log 2>&1 &
./core/run.sh python core/audit.py fmri_connectivity "autism_abide adhd_adhd200" "1 2 3" > logs/audit_fmri_connectivity.log 2>&1 &
./core/run.sh python core/audit.py eeg_psych "mdd_tdbrain adhd_tdbrain sz_aseeg abnormal_tuab" "1 2 3" > logs/audit_eeg_psych.log 2>&1 &
./core/run.sh python core/audit.py neonatal_seizure "helsinki" "1 2 3" > logs/audit_neonatal_seizure.log 2>&1 &
wait
