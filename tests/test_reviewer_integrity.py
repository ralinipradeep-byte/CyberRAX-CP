"""Additional independent reproducibility and provenance integrity checks."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from simulate import generate_dataset
from preprocess import split_ranges, _impute_by_split, build_features
from fusion import PlattCalibrator, select_threshold, adaptive_fusion
from metrics import classification_metrics


def test_validation_and_test_changes_do_not_modify_training_features():
    data=generate_dataset(seed=42,n=600,train_end=360,validation_end=480)
    ranges=split_ranges(600,360,480)
    imp1, fl1,_=_impute_by_split(data,ranges)
    feat1,names1,_,_=build_features(data,imp1,fl1,ranges,window=30)
    changed=data.copy()
    changed.loc[360:,'pressure_in']=changed.loc[360:,'pressure_in']+50000.0
    changed.loc[360:,'flow_main']=changed.loc[360:,'flow_main']-20000.0
    imp2, fl2,_=_impute_by_split(changed,ranges)
    feat2,names2,_,_=build_features(changed,imp2,fl2,ranges,window=30)
    assert names1==names2
    pd.testing.assert_frame_equal(feat1.iloc[:360],feat2.iloc[:360])


def test_platt_calibration_uses_only_supplied_validation_observations():
    yv=np.array([0,0,0,1,1,1,0,1],dtype=int)
    pv=np.array([.05,.2,.3,.75,.8,.9,.15,.7])
    model=PlattCalibrator().fit(pv,yv)
    q=model.predict([.2,.8])
    assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all()
    assert q[0]<q[1]
    t,_=select_threshold(yv,pv)
    assert 0<=t<=1


def test_fused_probability_and_weights_are_probability_distributions():
    rng=np.random.default_rng(91)
    a=rng.uniform(0,1,1000);b=rng.uniform(0,1,1000)
    f,wa,wb,_,_=adaptive_fusion(a,b,12.3,38.7)
    assert np.allclose(wa+wb,1.0,atol=1e-10)
    assert ((f>=0)&(f<=1)).all()


def test_materialized_data_partition_and_feature_invariants_if_present():
    p=ROOT/'data'/'processed'
    if not (p/'window_index.csv').exists(): pytest.skip('processed arrays not distributed in source-only checkout')
    m=pd.read_csv(p/'window_index.csv')
    assert m.groupby('split').size().to_dict()=={'train':14371,'validation':4771,'test':4771}
    bounds={'train':(0,14400),'validation':(14400,19200),'test':(19200,24000)}
    for split,(a,b) in bounds.items():
        v=m[m.split==split]
        assert (v.global_start_index>=a).all() and (v.global_end_index<b).all()
        assert (v.global_end_index-v.global_start_index==29).all()
    names=json.loads((p/'domain_feature_columns.json').read_text())
    assert len(names)==69 and len(set(names))==69
    y=np.load(p/'y_windows_int8.npy')
    assert len(y)==len(m) and np.array_equal(y,m.label.to_numpy())


def test_saved_quick_predictions_metrics_recompute_if_present():
    p=ROOT/'results'/'quick10_core_verified'
    if not (p/'metrics_by_seed.csv').exists() or not (p/'seed_11'/'predictions_test.csv').exists():
        pytest.skip('full quick predictions not distributed in source-only checkout')
    metrics=pd.read_csv(p/'metrics_by_seed.csv')
    assert metrics.seed.nunique()==10
    for seed in sorted(metrics.seed.unique()):
        folder=p/f'seed_{seed}'
        predictions=pd.read_csv(folder/'predictions_test.csv')
        meta=json.loads((folder/'run_metadata.json').read_text())
        assert meta['mode']=='quick'
        assert len(predictions)==4771
        assert np.allclose(predictions.weight_temporal+predictions.weight_domain,1.0,atol=1e-6)
        actual=classification_metrics(predictions.y_true,predictions.p_cyberrax_cp,threshold=meta['thresholds_validation_only']['CyberRAX-CP'])
        reported=metrics[(metrics.seed==seed)&(metrics.model=='CyberRAX-CP')].iloc[0]
        for k in ('f1','roc_auc','ece','brier','precision','recall'):
            assert abs(float(actual[k])-float(reported[k]))<1e-6,(seed,k)
