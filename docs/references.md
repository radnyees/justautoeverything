# References — datasets, baselines and resources

Every dataset and published number used to plan and evaluate the projects. Access status as
checked on 2026-10-11/12. "Baseline" = the published result each project is compared against;
all of our numbers are subject/patient-independent with a held-out evaluation set scored once.

## 1. Resting-state EEG → psychiatric disorders

**Datasets**
| Dataset | What | Access | Link |
|---|---|---|---|
| Mumtaz et al. 2016 MDD EEG | 34 MDD + 30 healthy, 19 ch, 256 Hz, eyes open/closed + task; 0.9 GB, CC BY 4.0 | open | https://figshare.com/articles/dataset/EEG_Data_New/4244171 |
| OpenNeuro ds003478 (Cavanagh, "EEG: Depression rest") | 122 students selected on high/low BDI (not clinical diagnoses), 64 ch, 10.6 GB, CC0 | open | https://openneuro.org/datasets/ds003478 |
| MODMA (Lanzhou) | 24 MDD + 29 HC, 128 ch resting | application + EULA | https://modma.lzu.edu.cn/data/application/ |
| TDBRAIN (van Dijk et al. 2022, *Sci Data*) | 1,274 patients (426 MDD, 271 ADHD, 75 OCD, 47 healthy), 26 ch, 500 Hz, 15 GB | ORCID login + DUA | https://brainclinics.com/resources/tdbrain-dataset · paper https://pmc.ncbi.nlm.nih.gov/articles/PMC9198070/ |
| Healthy Brain Network EEG (OpenNeuro ds005505–ds005516) | thousands of children, 129 ch; open data has only CBCL scores, diagnoses need the HBN DUA | open / DUA | https://openneuro.org/datasets/ds005505 |
| IEEE DataPort "EEG data for ADHD / Control children" | 61 ADHD + 60 controls, 19 ch, 128 Hz, visual-attention TASK | free IEEE account | https://ieee-dataport.org/open-access/eeg-data-adhd-control-children |
| ASEEG schizophrenia (2026) | 51 SZ + 50 HC, 5 min EO + 5 min EC, CC BY 4.0 | open | https://zenodo.org/records/18029536 |
| MSU adolescents (Borisov et al.) | 45 SZ + 39 HC boys, 16 ch, 128 Hz, 1 min EC | open (no license stated) | http://brain.bio.msu.ru/eeg_schizophrenia.htm |
| RepOD (Olejarczyk & Jernajczyk 2017) | 14 SZ + 14 HC, 19 ch, 250 Hz, CC0 | open | https://doi.org/10.18150/repod.0107441 |
| TUAB (TUH Abnormal EEG Corpus v3.0.1) | 2,993 recordings (train 2,717 / eval 276) | NEDC credentials (same as TUSZ) | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |

**Baselines / literature**
- MDD, ds003478, subject-wise CV: segment-level AUC 0.936 — https://pmc.ncbi.nlm.nih.gov/articles/PMC13269828/
- MDD, MODMA, subject-independent 5-fold: AUC 0.899, acc 84.8% — https://doi.org/10.3390/bioengineering13030358
- ADHD (IEEE set), subject-independent CNN: acc 72.7%, AUC 0.72; SVM drops 90.0% → 65.6% without leakage — https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1925266/full
- Schizophrenia, leakage-free: MSU 81.3% (vs 85.3% leaky), RepOD 58.1% (vs 86.3% leaky); ~65% of published SZ-EEG papers leak — Racz & Csukly, https://www.medrxiv.org/content/10.64898/2025.12.04.25341680v2.full.pdf
- TUAB, official patient-disjoint eval: 81–86% accuracy — Gemein et al. 2020, https://arxiv.org/abs/2002.05115

## 2. fMRI functional connectivity → autism / ADHD

**Datasets**
| Dataset | What | Access | Link |
|---|---|---|---|
| ABIDE Preprocessed (Preprocessed Connectomes Project) | 1,035 subjects (505 ASD / 530 TD, 20 sites); CPAC/CCS/DPARSF/NIAK; ROI time series for AAL, CC200, CC400, Dosenbach160, EZ, HO, TT; CC BY-NC-SA | open S3 | http://preprocessed-connectomes-project.org/abide/ |
| ADHD-200 Preprocessed (Athena / NIAK) | 947 subjects (362 ADHD / 585 TD, 8 sites); time series for AAL, CC200, CC400, EZ, HO, TT | free NITRC login | http://preprocessed-connectomes-project.org/adhd200/ · https://www.nitrc.org/frs/?group_id=383 |
| Schaefer 2018 parcellation | 100–1000 ROIs (via nilearn), MIT | open | https://github.com/ThomasYeoLab/CBIG |

**Baselines**
- Abraham et al. 2017, *NeuroImage* — ABIDE autism ~67% accuracy incl. unseen sites: https://doi.org/10.1016/j.neuroimage.2016.10.045
- Heinsfeld et al. 2018, *NeuroImage: Clinical* — 70% (10-fold), 65% leave-site-out: https://doi.org/10.1016/j.nicl.2017.08.017
- ADHD-200 competition (Brown et al. 2012, *Front Syst Neurosci*) — best imaging 60.5% (3-way); phenotype-only 62.5%: https://pmc.ncbi.nlm.nih.gov/articles/PMC3460316/

## 3. Neonatal seizure detection

| Dataset | What | Access | Link |
|---|---|---|---|
| Helsinki neonatal EEG (Stevenson et al. 2019, *Sci Data*) | 79 term neonates (39 with seizures by consensus), 19 ch, 256 Hz, ~74 min median, 3 expert annotations; 4.3 GB, CC BY 4.0 | open | https://zenodo.org/records/2547147 |

**Baselines**
- Hogan et al. 2025 — ConvNeXt trained on Cork, tested on Helsinki: concatenated AUC 0.982; SVM (Tapani) leave-one-out 0.955: https://pmc.ncbi.nlm.nih.gov/articles/PMC11711471/
- Tapani et al. 2022 — 10-fold median AUC 0.992, concatenated 0.956: https://arxiv.org/abs/2202.12023
- O'Shea et al. 2020 — FCN trained on Cork, 95.6% AUC on Helsinki: https://arxiv.org/abs/2105.13854

## 4. Gut microbiome (16S) → colorectal cancer / IBD

| Dataset | What | Access | Link |
|---|---|---|---|
| MicrobiomeHD (Duvallet et al. 2017, *Nat Commun*) | 28 case–control 16S studies, processed de novo OTU tables + RDP taxonomy; CC BY-NC 4.0. Used: CRC — Baxter 2016 (120 CRC / 172 H), Zeller 2014, Zackular 2014, Xiang, Zhao; IBD — Gevers 2014 RISK (700 CD / 340 non-IBD samples, 697 subjects), Alm, Engstrand, Huttenhower | open | https://doi.org/10.5281/zenodo.840333 · paper https://doi.org/10.1038/s41467-017-01973-8 |
| ML Repo (Vangay et al. 2019, *GigaScience*) | curated microbiome ML tasks | open | https://github.com/knights-lab/MLRepo |
| Raw reads for ASV reprocessing | Baxter PRJNA290926 (~8.5 GB); Gevers PRJNA237362 (~20 GB 16S) | open (ENA/SRA) | https://www.ebi.ac.uk/ena/browser/view/PRJNA290926 |

**Baselines**
- Duvallet 2017 (RF, genus, within-study 5-fold): CRC 0.77 (Baxter) / 0.82 (Zeller); IBD 0.71 (Gevers stool) — paper above.
- Sze & Schloss 2018, *mBio* — cross-study 16S CRC median AUC 0.71: https://pmc.ncbi.nlm.nih.gov/articles/PMC5989068/
- Wirbel et al. 2019, *Nat Med* (metagenomics) — CRC LOSO ≥ 0.83: https://doi.org/10.1038/s41591-019-0406-6
- Pasolli et al. 2016, *PLoS Comput Biol* (metagenomics) — CRC 0.87, IBD 0.89: https://doi.org/10.1371/journal.pcbi.1004977

## 5. Proteomics → Alzheimer's disease

| Dataset | What | Access | Link |
|---|---|---|---|
| PXD065835 (iProX) | PLASMA, Orbitrap Astral DIA; 53 AD + 36 controls; DIA-NN protein matrix | open | https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD065835 |
| PXD067951 | PLASMA, Proteograph + DIA-NN; 42 MCI + 13 controls; labels from the paper's supplement | open | https://www.ebi.ac.uk/pride/archive/projects/PXD067951 · paper https://doi.org/10.1038/s41598-026-47088-3 |
| Bader et al. 2020, *Mol Syst Biol* (PXD016278) | CSF DIA proteomics, 212 people across 4 sites; protein matrix + biomarker annotations | open | https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD016278 |
| ADNI | plasma 190-analyte panel (Myriad RBM), CSF SomaScan 7K | application + DUA (~2–3 weeks) | https://adni.loni.usc.edu/ |
| Bio-Hermes | plasma TMT-MS, 992 people, amyloid PET | AD Workbench DUA | https://pmc.ncbi.nlm.nih.gov/articles/PMC13299064/ |
| GNPC | ~35k SomaScan samples | AD Workbench application | https://www.nature.com/articles/s41591-025-03834-0 |

**Baselines**
- Bio-Hermes plasma MS, amyloid PET: RF/GBM AUROC 0.79–0.81 — PMC13299064 (above).
- Plasma p-tau217 for amyloid PET (cognitively normal): 0.85, combined 0.87 — https://link.springer.com/article/10.1186/s13195-026-02080-x
- DIA-MS plasma 10-protein panel for CSF biomarker positivity: 0.91 — https://www.medrxiv.org/content/10.64898/2026.05.26.26353866v1
- Bader 2020 CSF classifier AUC 0.91 (uses tau, which defines the label — inflated).

## 6. Drug response (CCLE / PRISM)

| Resource | What | Access | Link |
|---|---|---|---|
| DepMap 24Q4 CCLE expression | `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, 1,673 cell lines × 19,193 genes, CC BY 4.0 | open (figshare) | https://doi.org/10.25452/figshare.plus.27993248 |
| PRISM Repurposing secondary screen 19Q4 (Corsello et al. 2020, *Nat Cancer*) | dose–response AUC, 1,448 drugs × 480 cell lines, CC BY 4.0 | open | https://figshare.com/articles/dataset/PRISM_Repurposing_19Q3_Primary_Screen/9393293 · paper https://pmc.ncbi.nlm.nih.gov/articles/PMC7328899/ |
| GDSC1/GDSC2 (release 8.4) | external validation drug response | free for research | https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/ |
| MSigDB 2026.1 / Reactome / STRING v12.5 | pathway gene sets and PPI network (CC BY 4.0 / CC0 / CC BY 4.0) | open | https://www.gsea-msigdb.org · https://reactome.org · https://string-db.org |

**Baselines**
- Carli et al. 2025 — GDSC median per-drug Pearson 0.40 on held-out cell lines (286 drugs): https://pmc.ncbi.nlm.nih.gov/articles/PMC11828915/
- Corsello et al. 2020 — 38 PRISM secondary-screen drugs with predictability > 0.4: paper above.

## Method lineage
- Karpathy, *autoresearch* (one editable file, one metric, keep-if-better LLM loop): https://github.com/karpathy/autoresearch
- autoepi (this author's EEG seizure-detection loop; protocol reused here): https://github.com/radnyees/autoepi
