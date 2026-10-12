# Archived: ibd_gevers (stool only) -- not reported

MicrobiomeHD's processed Gevers 2014 OTU table contains only the 162 STOOL samples
(146 Crohn's / 16 controls), not the 1,040 biopsy+stool samples listed in its metadata. With
16 controls, each held-out Eval set (25% of subjects) had only 4 controls, so Eval AUROC and its
confidence interval are meaningless (e.g. 95% CI -0.44..+0.28). The task was replaced by
`ibd_pooled` (four IBD stool studies pooled). These runs are kept only as a record.
