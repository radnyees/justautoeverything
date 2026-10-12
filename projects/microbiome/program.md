# Project: gut microbiome 16S -> disease (colorectal cancer, Crohn's disease)

## Data you receive
- `items`: pandas DataFrame of RAW 16S OTU read counts, rows = samples, columns = OTUs.
  Column names are RDP taxonomy strings `k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Blautia;s__;d__denovo123`
  (levels can be empty, e.g. `g__;`). OTU ids (`d__denovoN`) are de novo and study-specific;
  taxonomy names are comparable across studies. Sequencing depth varies a lot between samples.
- `ctx["meta"]` / `ctx["meta_train"]` / `ctx["meta_test"]`: DataFrame with `sample_location`
  (stool / Terminal Ileum / Rectum) and `study`.
- Predictions: one float per sample, higher = more likely diseased. Metric: AUROC.

## Open questions worth exploring
Normalization (rarefaction vs relative abundance vs CLR / other log-ratios, pseudocounts),
taxonomic level (OTU vs genus vs family, or several), prevalence filtering, diversity metrics
(Shannon, richness, Faith-like taxonomy-aware measures), compositional-data methods,
known disease-associated taxa (e.g. Fusobacterium, Peptostreptococcus, Porphyromonas, Parvimonas
for CRC; loss of Faecalibacterium / Roseburia and gain of Enterobacteriaceae / Fusobacteriaceae
for IBD), co-abundance groups, classifiers and their regularization, and how to make a model
that transfers to other studies (different sequencing depth, region, OTU sets).
