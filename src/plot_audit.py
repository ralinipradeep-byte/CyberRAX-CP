#!/usr/bin/env python3
"""Reviewer diagnostic plots computed strictly from executed quick-check predictions."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',default='results/quick10_core_verified')
    ap.add_argument('--out',default='figures/quick10_DIAGNOSTIC')
    args=ap.parse_args()
    inp=Path(args.results);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(inp/'metrics_by_seed.csv')
    p=pd.read_csv(inp/'predictions_test_all_seeds.csv')
    base=p[p.seed==p.seed.min()].copy()
    y=base.y_true.to_numpy();f=base.p_cyberrax_cp.to_numpy()
    fig,ax=plt.subplots(figsize=(8,4.7))
    for name in ['CyberRAX-CP','Fixed-weight fusion','GRU-Attention','XGBoost']:
        subset=d[d.model==name].sort_values('seed')
        ax.plot(subset.seed,subset.f1,marker='o',label=name)
    ax.set(title='10-seed quick-check F1 (not manuscript reproduction)',xlabel='Training seed',ylabel='Test F1',ylim=(0,1))
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'01_seedwise_f1.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.7))
    for name in ['CyberRAX-CP','Fixed-weight fusion','GRU-Attention','XGBoost']:
        subset=d[d.model==name].sort_values('seed')
        ax.plot(subset.seed,subset.ece,marker='o',label=name)
    ax.set(title='10-seed quick-check calibration error',xlabel='Training seed',ylabel='ECE (15 bins)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'02_seedwise_ece.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));fpr,tpr,_=roc_curve(y,f);ax.plot(fpr,tpr,label='Seed 11');ax.plot([0,1],[0,1],'--')
    ax.set(title='Executed quick-check ROC — seed 11',xlabel='False positive rate',ylabel='True positive rate')
    ax.legend();fig.tight_layout();fig.savefig(out/'03_quick_ROC.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));pr,re,_=precision_recall_curve(y,f);ax.plot(re,pr)
    ax.set(title='Executed quick-check precision–recall — seed 11',xlabel='Recall',ylabel='Precision')
    fig.tight_layout();fig.savefig(out/'04_quick_PR.png',dpi=220);plt.close(fig)
    pred_bins, true_bins=calibration_curve(y,f,n_bins=15,strategy='uniform')
    fig,ax=plt.subplots(figsize=(6,5));ax.plot(pred_bins,true_bins,marker='o');ax.plot([0,1],[0,1],'--')
    ax.set(title='Quick-check reliability diagram — seed 11',xlabel='Mean predicted probability',ylabel='Observed attack fraction')
    fig.tight_layout();fig.savefig(out/'05_reliability_diagram.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.4));ax.hist(base.weight_temporal,bins=25)
    ax.set(title='Quick-check adaptive temporal weights — seed 11',xlabel='Temporal branch weight',ylabel='Test windows')
    fig.tight_layout();fig.savefig(out/'06_temporal_weight_histogram.png',dpi=220);plt.close(fig)
    from json import loads
    thresholds=loads((inp/'seed_11'/'run_metadata.json').read_text())['thresholds_validation_only']
    cm=confusion_matrix(y,f>=thresholds['CyberRAX-CP'],labels=[0,1])
    fig,ax=plt.subplots(figsize=(5.4,4.5));im=ax.imshow(cm)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['normal','attack'],yticklabels=['normal','attack'],xlabel='Predicted',ylabel='True',title='Quick-check confusion matrix — seed 11')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'07_confusion_matrix.png',dpi=220);plt.close(fig)
    a=pd.concat([pd.read_csv(q).assign(seed=int(q.parent.name.split('_')[-1])) for q in inp.glob('seed_*/attack_specific_recall.csv')])
    b=a.groupby('attack_type').recall.agg(['mean','std']).sort_values('mean',ascending=True)
    fig,ax=plt.subplots(figsize=(9,4.7));ax.barh([x.replace('_',' ') for x in b.index],b['mean'],xerr=b['std'].fillna(0));ax.set(xlim=(0,1),xlabel='Mean recall across seeds (normal row = FPR)',title='Quick-check attack-type outcomes')
    fig.tight_layout();fig.savefig(out/'08_attack_type_recall.png',dpi=220);plt.close(fig)
    rob=pd.read_csv(inp/'robustness'/'robustness_by_seed.csv')
    order=['clean','5pct_gaussian_noise','10pct_missing_telemetry','combined_noise_and_missingness']
    b=rob.groupby('condition').f1.agg(['mean','std']).reindex(order)
    fig,ax=plt.subplots(figsize=(8.2,4.7));ax.bar(range(len(b)),b['mean'],yerr=b['std'],capsize=4)
    ax.set(xticks=range(len(b)),xticklabels=['Clean','5% Gaussian noise','10% missing','Combined'],ylim=(0,1),ylabel='F1',title='Executed quick-check perturbation outcomes')
    fig.tight_layout();fig.savefig(out/'09_robustness.png',dpi=220);plt.close(fig)
    print('Created',len(list(out.glob('*.png'))),'diagnostic images in',out)
if __name__=='__main__':main()
