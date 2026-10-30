#!/usr/bin/env python3
"""Audit measured quick-check outputs against the reference claims without conflating protocols."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result-dir',default='results/quick10_core_verified')
    ap.add_argument('--reference',default='data/metadata/manuscript_reported_results.csv')
    ap.add_argument('--out',default='docs')
    args=ap.parse_args();r=Path(args.result_dir);o=Path(args.out);o.mkdir(parents=True,exist_ok=True)
    reported=pd.read_csv(args.reference);measured=pd.read_csv(r/'metrics_summary.csv')
    mapping={'F1':'f1','ROC_AUC':'roc_auc','ECE':'ece','Brier':'brier','precision':'precision','recall':'recall','accuracy':'accuracy'}
    rows=[]
    for _,ref in reported.iterrows():
        if ref['metric'] not in mapping:continue
        key=mapping[ref['metric']]
        values=measured[(measured.model==ref.model_or_condition)&(measured.metric==key)]
        if values.empty:continue
        numeric=float(ref.value)
        if ref.unit=='percent':numeric/=100.0
        actual=float(values.iloc[0]['mean'])
        rows.append({'manuscript_table':ref.section_or_table,'model':ref.model_or_condition,'metric':key,
                     'manuscript_reference':numeric,'quick10_observed':actual,'difference_quick_minus_manuscript':actual-numeric,
                     'directly_comparable':False,
                     'reason':'Reconstructed raw data, 1 GRU epoch, 30 XGBoost trees; not original full-hyperparameter experiment'})
    pd.DataFrame(rows).to_csv(o/'manuscript_vs_quick10_DIAGNOSTIC_NOT_EQUIVALENT.csv',index=False,float_format='%.8f')
    summary=measured[(measured.model=='CyberRAX-CP') & measured.metric.isin(['f1','roc_auc','ece','brier'])]
    audit={'status':'NOT VERIFIED FOR MANUSCRIPT EQUIVALENCE','observations':24000,'quick_seeds':10,
           'original_GP_OTSim_raw_data_supplied':False,'original_run_logs_supplied':False,
           'full_hyperparameter_ten_seed_run_executed':False,
           'quick_mode_executed':True,'quick_mode_GRU_epochs':1,'quick_mode_XGBoost_estimators':30,
           'quick_mode_transformer_and_optional_baselines_evaluated':False,
           'reported_values_are_reference_only':True,
           'quick_results':{row.metric:{'mean':row['mean'],'sd':row['sd']} for _,row in summary.iterrows()}}
    (o/'reviewer_verification_status.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
