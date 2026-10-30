#!/usr/bin/env python3
"""Apply prespecified perturbations to trained branch models; calibration stays frozen."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from scipy.special import expit, logit
from models import GRUAttention
from metrics import classification_metrics
from fusion import adaptive_fusion
from train import load_data, predict_torch


def calibrate(p, spec):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return expit(float(spec["coef"][0])*logit(p) + float(spec["intercept"][0]))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="data/processed"); ap.add_argument("--results-dir",default="results/full"); ap.add_argument("--out",default=None)
    args=ap.parse_args(); data_dir=Path(args.data_dir); results=Path(args.results_dir); out=Path(args.out) if args.out else results/"robustness"
    out.mkdir(parents=True,exist_ok=True)
    meta, x_seq, x_dom, x_dom_scaled, y, splits = load_data(data_dir)
    tr, te = splits["train"], splits["test"]
    xte_seq = x_seq[te].copy(); xte_dom = x_dom[te].copy(); yte = y[te]
    dom_mean = x_dom[tr].mean(axis=0); dom_std = np.maximum(x_dom[tr].std(axis=0), 1e-6)
    rows=[]
    for run_dir in sorted(results.glob("seed_*")):
        if not run_dir.is_dir() or not (run_dir/"run_metadata.json").exists(): continue
        run_meta=json.loads((run_dir/"run_metadata.json").read_text(encoding="utf-8"))
        seed=int(run_meta["seed"]); rng=np.random.default_rng(seed+991)
        ckpt=torch.load(run_dir/"gru_attention.pt", map_location="cpu", weights_only=False)
        model=GRUAttention(input_size=xte_seq.shape[-1],hidden_size=64,num_layers=2,dropout=.2); model.load_state_dict(ckpt["state_dict"]); model.eval()
        tree=joblib.load(run_dir/"xgboost_domain_69.joblib")
        rel_t=run_meta["branch_reliability_temporal"]["reliability"]; rel_d=run_meta["branch_reliability_domain"]["reliability"]
        threshold=float(run_meta["thresholds_validation_only"]["CyberRAX-CP"])
        cal_t=run_meta["platt_temporal"]; cal_d=run_meta["platt_domain"]
        conditions={"clean":(xte_seq.copy(),xte_dom.copy())}
        xn=xte_seq.copy()+rng.normal(0,.05,xte_seq.shape)
        dn=xte_dom.copy()+rng.normal(0,.05*dom_std,xte_dom.shape)
        conditions["5pct_gaussian_noise"]=(xn,dn)
        xm=xte_seq.copy(); dm=xte_dom.copy()
        mask_x=rng.random(xm.shape)<.10; xm[mask_x]=0.0
        mask_d=rng.random(dm.shape)<.10; dm[mask_d]=np.broadcast_to(dom_mean,dm.shape)[mask_d]
        conditions["10pct_missing_telemetry"]=(xm,dm)
        xc=xte_seq.copy()+rng.normal(0,.05,xte_seq.shape); dc=xte_dom.copy()+rng.normal(0,.05*dom_std,xte_dom.shape)
        mask_x=rng.random(xc.shape)<.10; xc[mask_x]=0.0
        mask_d=rng.random(dc.shape)<.10; dc[mask_d]=np.broadcast_to(dom_mean,dc.shape)[mask_d]
        conditions["combined_noise_and_missingness"]=(xc,dc)
        for condition,(xx,dd) in conditions.items():
            p_t=calibrate(predict_torch(model,xx),cal_t)
            p_d=calibrate(tree.predict_proba(dd)[:,1],cal_d)
            p_f,wt,wd,_,_=adaptive_fusion(p_t,p_d,rel_t,rel_d)
            metrics=classification_metrics(yte,p_f,threshold=threshold)
            rows.append({"seed":seed,"condition":condition,"mean_weight_temporal":float(np.mean(wt)),
                         "mean_weight_domain":float(np.mean(wd)),"sd_weight_temporal":float(np.std(wt,ddof=1)),**metrics})
        print(f"robustness completed for seed {seed}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/"robustness_by_seed.csv",index=False,float_format="%.8f")
    if not df.empty:
        summary=df.groupby("condition")[["f1","ece","brier","fpr","mean_weight_temporal","mean_weight_domain","sd_weight_temporal"]].agg(["mean","std"])
        summary.to_csv(out/"robustness_summary.csv",float_format="%.8f")
    (out/"README.txt").write_text("Perturbations are applied after training. Calibrators, branch reliabilities, and decision threshold are frozen from validation; test data are not used for refitting. Missing standardized temporal values are set to 0 (training mean), and missing domain values to training-window feature means.\n",encoding="utf-8")
    print(f"Wrote robustness results to {out.resolve()}")

if __name__=="__main__": main()
