#!/usr/bin/env python3
"""Create publication-style figures from generated data and actual experiment outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def dataset_figures(raw_csv: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = df["attack_type"].value_counts().reindex(["normal","false_data_injection","replay","command_injection","denial_of_service","scanning","man_in_the_middle","device_spoofing"], fill_value=0)
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts))); ax.set_xticklabels([x.replace("_", "\n") for x in counts.index], fontsize=8)
    ax.set_ylabel("Observations"); ax.set_title("Generated GP-OTSim-style label distribution")
    fig.tight_layout(); fig.savefig(out/"dataset_label_distribution.png", dpi=300); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, col in zip(axes, ["pressure_in","pressure_mid","pressure_out"]):
        ax.plot(df["t_seconds"].iloc[:2400], df[col].iloc[:2400], linewidth=.7)
        ax.set_ylabel(col.replace("_", " "))
    axes[-1].set_xlabel("Time (s)"); fig.suptitle("Example process telemetry (first 2,400 seconds)")
    fig.tight_layout(); fig.savefig(out/"sample_process_telemetry.png", dpi=300); plt.close(fig)


def result_figures(results_dir: Path, out: Path):
    metrics_file = results_dir/"metrics_by_seed.csv"
    preds_file = results_dir/"predictions_test_all_seeds.csv"
    if not metrics_file.exists():
        print(f"No metrics file at {metrics_file}; dataset figures only."); return
    m = pd.read_csv(metrics_file)
    order = [x for x in ["SVM-RBF","Random Forest","XGBoost","GRU-Attention","Transformer","Fixed-weight fusion","CyberRAX-CP"] if x in set(m.model)]
    summary = m.groupby("model")["f1"].agg(["mean","std"]).reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), capsize=3)
    ax.set_ylim(0, 1); ax.set_ylabel("F1-score"); ax.set_title("Test F1 across executed seeds")
    ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(out/"executed_f1_comparison.png", dpi=300); plt.close(fig)
    summary2 = m.groupby("model")[["ece","brier"]].mean().reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5)); x=np.arange(len(summary2)); width=.36
    ax.bar(x-width/2, summary2.ece, width, label="ECE"); ax.bar(x+width/2, summary2.brier, width, label="Brier score")
    ax.set_xticks(x); ax.set_xticklabels(summary2.index, rotation=25, ha="right"); ax.set_ylabel("Error (lower is better)")
    ax.set_title("Calibration metrics from executed predictions"); ax.legend(); fig.tight_layout()
    fig.savefig(out/"executed_calibration_comparison.png", dpi=300); plt.close(fig)
    if preds_file.exists():
        p = pd.read_csv(preds_file)
        p = p[p.seed == p.seed.min()]
        if "p_cyberrax_cp" in p.columns:
            y=p.y_true.to_numpy(); prob=p.p_cyberrax_cp.to_numpy()
            fpr,tpr,_=roc_curve(y,prob); prec,rec,_=precision_recall_curve(y,prob)
            fig,ax=plt.subplots(figsize=(5.5,4.5)); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],linestyle="--")
            ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.set_title("ROC curve (first executed seed)")
            fig.tight_layout(); fig.savefig(out/"executed_roc_curve.png",dpi=300); plt.close(fig)
            fig,ax=plt.subplots(figsize=(5.5,4.5)); ax.plot(rec,prec)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–recall curve (first executed seed)")
            fig.tight_layout(); fig.savefig(out/"executed_precision_recall_curve.png",dpi=300); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="data/raw/gpotsim_reconstructed.csv")
    ap.add_argument("--results",default="results/full"); ap.add_argument("--out",default="figures/full")
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    dataset_figures(Path(args.data),out); result_figures(Path(args.results),out)
    print(f"Figures saved to {out.resolve()}")

if __name__=="__main__": main()
