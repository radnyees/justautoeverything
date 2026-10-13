"""
projects/drug_response/data/build.py -- CCLE expression + PRISM drug response + pathway resources
-> /data/drug_response/.

  expression : DepMap 24Q4 OmicsExpressionProteinCodingGenesTPMLogp1 (log2(TPM+1)), CC BY 4.0
  models     : DepMap 24Q4 Model.csv (lineage etc.)
  response   : PRISM Repurposing 19Q4 SECONDARY screen dose-response AUC (Corsello et al. 2020),
               averaged over screens per (drug, cell line); CC BY 4.0
  gene sets  : MSigDB Hallmark (CC BY 4.0), Reactome pathways (CC0); STRING v12.5 human PPI
               (CC BY 4.0, combined score >= 700)
    python projects/drug_response/data/build.py /data/drug_response
"""
import gzip
import io
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL = {
    "expr": "https://ndownloader.figshare.com/files/51065489",
    "model": "https://ndownloader.figshare.com/files/51065297",
    "prism": "https://ndownloader.figshare.com/files/20237739",
    "hallmark": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt",
    "reactome": "https://reactome.org/download/current/ReactomePathways.gmt.zip",
    "string": "https://stringdb-downloads.org/download/protein.links.v12.5/9606.protein.links.v12.5.txt.gz",
    "string_info": "https://stringdb-downloads.org/download/protein.info.v12.5/9606.protein.info.v12.5.txt.gz",
}


def get(key):
    for t in range(5):
        try:
            req = urllib.request.Request(URL[key], headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=1800).read()
        except Exception as e:
            print("retry", key, e, flush=True)
    raise RuntimeError(key)


def main(out):
    os.makedirs(out, exist_ok=True)
    expr = pd.read_csv(io.BytesIO(get("expr")), index_col=0)
    expr = expr.loc[:, [c for c in expr.columns if "(" in c]]
    expr.columns = [c.split(" (")[0] for c in expr.columns]          # "TP53 (7157)" -> "TP53"
    expr = expr.loc[:, ~expr.columns.duplicated()].astype(np.float32)
    models = pd.read_csv(io.BytesIO(get("model")))
    pr = pd.read_csv(io.BytesIO(get("prism")), low_memory=False)
    pr = pr.dropna(subset=["auc", "depmap_id", "broad_id"])
    auc = pr.groupby(["depmap_id", "broad_id"])["auc"].mean().unstack()
    drugs = pr.drop_duplicates("broad_id").set_index("broad_id")[["name", "moa", "target", "phase", "disease.area", "indication"]]
    lines = sorted(set(auc.index) & set(expr.index))
    auc, expr = auc.loc[lines], expr.loc[lines]
    models = models.set_index("ModelID").reindex(lines)
    expr.to_parquet(f"{out}/expression.parquet")
    auc.astype(np.float32).to_parquet(f"{out}/prism_auc.parquet")
    drugs.loc[auc.columns].to_parquet(f"{out}/drugs.parquet")
    models[["OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype", "Sex", "Age"]].astype(str).to_parquet(f"{out}/models.parquet")
    print("lines", len(lines), "genes", expr.shape[1], "drugs", auc.shape[1],
          "missing", round(float(auc.isna().mean().mean()), 3), flush=True)

    os.makedirs(f"{out}/genesets", exist_ok=True)
    open(f"{out}/genesets/hallmark.gmt", "wb").write(get("hallmark"))
    z = zipfile.ZipFile(io.BytesIO(get("reactome")))
    open(f"{out}/genesets/reactome.gmt", "wb").write(z.read(z.namelist()[0]))
    info = pd.read_csv(io.BytesIO(gzip.decompress(get("string_info"))), sep="\t")
    name = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    links = pd.read_csv(io.BytesIO(gzip.decompress(get("string"))), sep=" ")
    links = links[links["combined_score"] >= 700]
    edges = pd.DataFrame({"a": links["protein1"].map(name), "b": links["protein2"].map(name),
                          "score": links["combined_score"].astype(np.int16)}).dropna()
    edges = edges[edges["a"] < edges["b"]]
    edges.to_parquet(f"{out}/genesets/string_v12.5_ge700.parquet")
    print("string edges", len(edges), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
