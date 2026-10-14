# Archived: ibd_pooled v2 (contained biopsies and repeated/related samples) -- not reported

Found in the pre-submission audit (2026-10-13, docs/revision2_checklist.md item B2). The v2 pooled
IBD task kept every sample of the four MicrobiomeHD studies regardless of body site and grouped by
sample whenever a study had no `subject` column:
* Huttenhower: 94 mucosal biopsies alongside 126 stool samples (biopsies cannot be linked to the
  same patient's stool sample by id);
* Engstrand (Willing 2010 twin study): 114 samples from 80 people (18 people with stool + ileum /
  descending-colon tissue) in 40 twin pairs; repeated and co-twin samples could fall on both sides
  of the Dev/Eval split;
* all samples were labelled "stool" in `meta`.
So the same person (or co-twin) could appear in Dev and Eval, and tissue type was mixed with
disease. v3 (current `prepare.py`): stool only (458 samples), grouped by person and by twin pair
in Engstrand. Loops, baselines, repeats and audit were rerun from scratch for v3.
Contents: runs/, runs_bestof20/, runs_optuna/, runs_repeat/ for ibd_pooled under v2.
