#!/usr/bin/env python3
"""Quantify cross-seed stability of domain-branch global feature rankings."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from metrics import explanation_stability


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results-dir",default="results/full"); ap.add_argument("--feature-names",default="data/processed/domain_feature_columns.json"); ap.add_argument("--out",default=None)
    args=ap.parse_args(); res=Path(args.results_dir); out=Path(args.out) if args.out else res/"explanation_stability"; out.mkdir(parents=True,exist_ok=True)
    names=json.loads(Path(args.feature_names).read_text(encoding="utf-8"))
    top_lists=[]; rank_vectors=[]; rows=[]
    for run_dir in sorted(res.glob("seed_*")):
        model_path=run_dir/"xgboost_domain_69.joblib"
        if not model_path.exists(): continue
        model=joblib.load(model_path); importance=np.asarray(model.feature_importances_,float)
        order=np.argsort(importance)[::-1]; top=[names[i] for i in order[:5]]
        ranks=np.empty_like(order); ranks[order]=np.arange(len(order))
        top_lists.append(top); rank_vectors.append(ranks.astype(float))
        for rank,idx in enumerate(order[:20],1): rows.append({"seed":int(run_dir.name.split("_")[-1]),"rank":rank,"feature":names[idx],"importance":float(importance[idx])})
    pd.DataFrame(rows).to_csv(out/"top20_features_by_seed.csv",index=False,float_format="%.8f")
    stability=explanation_stability(top_lists,rank_vectors)
    (out/"stability_summary.json").write_text(json.dumps(stability,indent=2),encoding="utf-8")
    print(json.dumps(stability,indent=2))

if __name__=="__main__": main()
