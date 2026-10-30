#!/usr/bin/env python3
"""Run smoke or full CyberRAX-CP experiments on preprocessed artifacts.

The full mode is intentionally explicit and can be compute-intensive (10 matched seeds).
"""
from __future__ import annotations
import argparse, json, os, random, time, platform, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from models import GRUAttention, TransformerDetector
from fusion import PlattCalibrator, fit_branch_reliability, adaptive_fusion, select_threshold
from metrics import classification_metrics, mean_sd_ci, paired_wilcoxon_table, explanation_stability


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(1)


def load_data(data_dir: Path):
    meta = pd.read_csv(data_dir / "window_index.csv")
    x_seq = np.load(data_dir / "windows_standardized_float64.npy", mmap_mode="r").astype(np.float32)
    x_dom = np.load(data_dir / "domain_features_float64.npy", mmap_mode="r").astype(np.float32)
    x_dom_scaled = np.load(data_dir / "domain_features_standardized_float64.npy", mmap_mode="r").astype(np.float32)
    y = np.load(data_dir / "y_windows_int8.npy").astype(np.int64)
    if not (len(meta) == len(x_seq) == len(x_dom) == len(x_dom_scaled) == len(y)):
        raise ValueError("Processed arrays and window_index.csv have different row counts")
    splits = {s: np.flatnonzero(meta.split.to_numpy() == s) for s in ["train", "validation", "test"]}
    return meta, x_seq, x_dom, x_dom_scaled, y, splits


def predict_torch(model, x, batch_size=256, return_attention=False):
    model.eval(); out=[]; att=[]
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(np.asarray(x[start:start+batch_size]), dtype=torch.float32).to(device)
            if return_attention and isinstance(model, GRUAttention):
                logits, weights = model(xb, return_attention=True); att.append(weights.cpu().numpy())
            else:
                logits = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(out) if out else np.array([])
    if return_attention:
        return p, (np.concatenate(att) if att else np.empty((0, 0)))
    return p


def train_torch(model_kind, x_train, y_train, x_val, y_val, seed, max_epochs, patience, batch_size, learning_rate, out_path=None):
    seed_everything(seed)
    if model_kind == "gru": model = GRUAttention(input_size=x_train.shape[-1], hidden_size=64, num_layers=2, dropout=.2)
    elif model_kind == "transformer": model = TransformerDetector(input_size=x_train.shape[-1], d_model=64, nhead=4, num_layers=2, dropout=.2, max_length=x_train.shape[1])
    else: raise ValueError(model_kind)
    requested_device = os.environ.get("CYBERRAX_DEVICE", "auto").strip().lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("CYBERRAX_DEVICE must be auto, cpu, or cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available())) else "cpu")
    model = model.to(device)
    y_train = np.asarray(y_train, dtype=np.float32); y_val = np.asarray(y_val, dtype=np.int64)
    pos = max(float(y_train.sum()), 1.0); neg = max(float(len(y_train)-y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if model_kind == "gru" else torch.optim.AdamW(model.parameters(), lr=learning_rate)
    ds = TensorDataset(torch.as_tensor(np.asarray(x_train), dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    best_f1, best_epoch, best_state, wait = -1.0, 0, None, 0
    history=[]
    for epoch in range(1, max_epochs+1):
        model.train(); loss_sum=0.0; n_seen=0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); optimizer.step()
            loss_sum += float(loss.detach())*len(xb); n_seen += len(xb)
        p_val = predict_torch(model, x_val)
        # Checkpoint selection uses the standard 0.50 validation threshold for speed and consistency.
        # The final operating threshold is selected separately on validation after model fitting.
        val_f1 = float(f1_score(y_val, p_val >= 0.5, zero_division=0))
        history.append({"epoch": epoch, "train_loss": loss_sum/max(n_seen,1), "validation_f1_at_0_5": val_f1, "checkpoint_threshold": 0.5})
        if val_f1 > best_f1 + 1e-7:
            best_f1, best_epoch, wait = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    if out_path:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "seed": seed, "model_kind": model_kind, "best_epoch": best_epoch,
                    "best_validation_f1": best_f1, "history": history}, out_path)
        pd.DataFrame(history).to_csv(out_path.with_suffix(".history.csv"), index=False)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_validation_f1": float(best_f1)}


def fit_xgb(x_train, y_train, x_val, y_val, seed, estimators=400, early_stopping=30):
    model = XGBClassifier(n_estimators=estimators, max_depth=5, learning_rate=.05, subsample=.8,
                          colsample_bytree=.9, objective="binary:logistic", eval_metric="logloss",
                          n_jobs=1, random_state=seed, reg_lambda=1.0, tree_method="hist",
                          early_stopping_rounds=early_stopping if estimators > early_stopping else None)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def fit_other_baselines(x_train_scaled, y_train, seed, smoke=False, rf_estimators=300, svm_max_train_samples=None):
    svm_x, svm_y = x_train_scaled, y_train
    if svm_max_train_samples is not None and len(svm_y) > int(svm_max_train_samples):
        from sklearn.model_selection import train_test_split
        sampled_idx, _ = train_test_split(np.arange(len(svm_y)), train_size=int(svm_max_train_samples),
                                          random_state=seed, stratify=svm_y)
        svm_x, svm_y = svm_x[sampled_idx], svm_y[sampled_idx]
    svm = SVC(C=10, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
    svm.fit(svm_x, svm_y)
    rf = RandomForestClassifier(n_estimators=20 if smoke else rf_estimators, max_depth=12, min_samples_leaf=2,
                                max_features="sqrt", class_weight="balanced_subsample", n_jobs=1, random_state=seed)
    rf.fit(x_train_scaled, y_train)
    return svm, rf


def _metrics_rows(seed, p_test_map, y_test, thresholds):
    rows=[]
    for model_name, p in p_test_map.items():
        m = classification_metrics(y_test, p, threshold=float(thresholds.get(model_name, .5)))
        rows.append({"seed": seed, "model": model_name, **m})
    return rows


def run_one_seed(seed, data, out_dir: Path, mode="full", cfg=None):
    meta, x_seq, x_dom, x_dom_scaled, y, splits = data
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    smoke = mode == "smoke"
    if smoke:
        # Use deterministic early samples for a quick code-path test; this is not a publication run.
        tr = tr[:min(2048, len(tr))]; va = va[:min(1024, len(va))]; te = te[:min(1024, len(te))]
        if np.unique(y[tr]).size < 2: tr = splits["train"]
        if np.unique(y[va]).size < 2: va = splits["validation"]
    seed_everything(seed)
    run_dir = out_dir / f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    xtr, xva, xte = x_seq[tr], x_seq[va], x_seq[te]
    dtr, dva, dte = x_dom[tr], x_dom[va], x_dom[te]
    dtr_s, dva_s, dte_s = x_dom_scaled[tr], x_dom_scaled[va], x_dom_scaled[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    seedspec = cfg.get("models", {}) if cfg else {}
    neural = seedspec.get("temporal", {})
    epochs = int(cfg.get("smoke", {}).get("max_epochs", 1)) if smoke else int(neural.get("max_epochs", 40))
    patience = 1 if smoke else int(neural.get("patience", 12))
    batch = 128 if smoke else int(neural.get("batch_size", 64))
    lr = float(neural.get("learning_rate", .001))
    model_file = run_dir / "gru_attention.pt"
    gru, gru_hist, gru_info = train_torch("gru", xtr, ytr, xva, yva, seed, epochs, patience, batch, lr, model_file)
    p_gru_val_raw = predict_torch(gru, xva)
    t_gru0 = time.perf_counter(); p_gru_test_raw, att_test = predict_torch(gru, xte, return_attention=True); gru_test_seconds = time.perf_counter()-t_gru0

    xgb_estimators = int(cfg.get("smoke", {}).get("xgb_estimators", 20)) if smoke else int(seedspec.get("xgboost", {}).get("n_estimators", 400))
    xgb = fit_xgb(dtr, ytr, dva, yva, seed, estimators=xgb_estimators, early_stopping=10 if smoke else 30)
    p_xgb_val_raw = xgb.predict_proba(dva)[:,1]
    t_xgb0 = time.perf_counter(); p_xgb_test_raw = xgb.predict_proba(dte)[:,1]; xgb_test_seconds = time.perf_counter()-t_xgb0
    joblib.dump(xgb, run_dir / "xgboost_domain_69.joblib")

    # Calibration is fitted on validation predictions only.
    cal_t = PlattCalibrator().fit(p_gru_val_raw, yva)
    cal_d = PlattCalibrator().fit(p_xgb_val_raw, yva)
    p_t_val, p_t_test = cal_t.predict(p_gru_val_raw), cal_t.predict(p_gru_test_raw)
    p_d_val, p_d_test = cal_d.predict(p_xgb_val_raw), cal_d.predict(p_xgb_test_raw)
    rel_t = fit_branch_reliability(yva, p_t_val, n_bins=15)
    rel_d = fit_branch_reliability(yva, p_d_val, n_bins=15)
    p_f_val, wt_val, wd_val, _, _ = adaptive_fusion(p_t_val, p_d_val, rel_t["reliability"], rel_d["reliability"])
    t_fusion0 = time.perf_counter(); p_f_test, wt_test, wd_test, ct_test, cd_test = adaptive_fusion(p_t_test, p_d_test, rel_t["reliability"], rel_d["reliability"]); fusion_test_seconds = time.perf_counter()-t_fusion0
    p_fixed_val = .5*p_t_val + .5*p_d_val; p_fixed_test = .5*p_t_test + .5*p_d_test
    thresholds = {}
    for name, p in {"CyberRAX-CP":p_f_val, "Fixed-weight fusion":p_fixed_val,
                    "GRU-Attention":p_gru_val_raw, "XGBoost":p_xgb_val_raw,
                    "Temporal branch only":p_t_val, "Domain-aware XGBoost":p_d_val}.items():
        thresholds[name], _ = select_threshold(yva, p)

    p_test_map = {
        "CyberRAX-CP": p_f_test,
        "XGBoost": p_xgb_test_raw,
        "GRU-Attention": p_gru_test_raw,
        "Temporal branch only": p_t_test,
        "Domain-aware XGBoost": p_d_test,
        "Fixed-weight fusion": p_fixed_test,
    }
    if not smoke and not seedspec.get("skip_optional_baselines", False):
        rf_estimators = int(seedspec.get("random_forest", {}).get("n_estimators", 300))
        svm_max = seedspec.get("svm", {}).get("max_train_samples", None)
        svm, rf = fit_other_baselines(dtr_s, ytr, seed, smoke=False, rf_estimators=rf_estimators, svm_max_train_samples=svm_max)
        p_svm_val, p_svm_test = svm.predict_proba(dva_s)[:,1], svm.predict_proba(dte_s)[:,1]
        p_rf_val, p_rf_test = rf.predict_proba(dva_s)[:,1], rf.predict_proba(dte_s)[:,1]
        thresholds["SVM-RBF"], _ = select_threshold(yva, p_svm_val)
        thresholds["Random Forest"], _ = select_threshold(yva, p_rf_val)
        p_test_map["SVM-RBF"], p_test_map["Random Forest"] = p_svm_test, p_rf_test
        transformer_cfg = seedspec.get("transformer", {})
        if not transformer_cfg.get("skip", False):
            trans, trans_hist, trans_info = train_torch(
                "transformer", xtr, ytr, xva, yva, seed,
                int(transformer_cfg.get("max_epochs", 40)), int(transformer_cfg.get("patience", 12)),
                int(transformer_cfg.get("batch_size", 64)), float(transformer_cfg.get("learning_rate", .0005)),
                run_dir / "transformer.pt")
            p_trans_val, p_trans_test = predict_torch(trans, xva), predict_torch(trans, xte)
            thresholds["Transformer"], _ = select_threshold(yva, p_trans_val)
            p_test_map["Transformer"] = p_trans_test
        # Generic feature ablation: first 55 documented base features only.
        generic = fit_xgb(dtr[:, :55], ytr, dva[:, :55], yva, seed, estimators=xgb_estimators, early_stopping=30)
        p_generic_val, p_generic_test = generic.predict_proba(dva[:, :55])[:,1], generic.predict_proba(dte[:, :55])[:,1]
        thresholds["Generic-feature XGBoost"], _ = select_threshold(yva, p_generic_val)
        p_test_map["Generic-feature XGBoost"] = p_generic_test
        joblib.dump(svm, run_dir / "svm_rbf.joblib"); joblib.dump(rf, run_dir / "random_forest.joblib")
        joblib.dump(generic, run_dir / "xgboost_generic_55.joblib")

    metric_rows = _metrics_rows(seed, p_test_map, yte, thresholds)
    for row in metric_rows:
        row["false_alarms_per_simulated_hour"] = float(row["fp"] / max(len(yte)/3600.0, 1e-9))
    test_meta = meta.iloc[te].reset_index(drop=True)
    predictions = pd.DataFrame({
        "seed": seed, "split": "test", "window_id": test_meta["window_id"].to_numpy(),
        "sample_id": test_meta["end_sample_id"].to_numpy(), "attack_type": test_meta["attack_type"].astype(str).to_numpy(), "y_true": yte,
        "p_temporal_raw": p_gru_test_raw, "p_temporal_calibrated": p_t_test,
        "p_domain_raw": p_xgb_test_raw, "p_domain_calibrated": p_d_test,
        "p_fused_adaptive": p_f_test, "p_fused_fixed": p_fixed_test,
        "weight_temporal": wt_test, "weight_domain": wd_test,
        "confidence_temporal": ct_test, "confidence_domain": cd_test,
    })
    for name, p in p_test_map.items():
        predictions["p_" + name.lower().replace(" ", "_").replace("-", "_")] = p
    predictions.to_csv(run_dir / "predictions_test.csv", index=False, float_format="%.8f")
    pred_attack = (p_f_test >= float(thresholds["CyberRAX-CP"])).astype(int)
    attack_rows = []
    for attack in sorted(set(test_meta["attack_type"].astype(str)) - {"normal"}):
        mask = test_meta["attack_type"].astype(str).to_numpy() == attack
        attack_rows.append({"seed": seed, "attack_type": attack, "n_attack_windows": int(mask.sum()),
                            "recall": float(np.mean(pred_attack[mask] == 1)) if mask.any() else float("nan")})
    normal_mask = test_meta["attack_type"].astype(str).to_numpy() == "normal"
    attack_rows.append({"seed": seed, "attack_type": "normal_false_positive_rate", "n_attack_windows": int(normal_mask.sum()),
                        "recall": float(np.mean(pred_attack[normal_mask] == 1)) if normal_mask.any() else float("nan")})
    pd.DataFrame(attack_rows).to_csv(run_dir / "attack_specific_recall.csv", index=False, float_format="%.8f")
    pd.DataFrame({"y_validation": yva, "p_temporal_raw": p_gru_val_raw, "p_temporal_calibrated": p_t_val,
                  "p_domain_raw": p_xgb_val_raw, "p_domain_calibrated": p_d_val,
                  "p_fused_adaptive": p_f_val, "p_fused_fixed": p_fixed_val,
                  "weight_temporal": wt_val, "weight_domain": wd_val}).to_csv(run_dir / "validation_calibration_audit.csv", index=False)
    run_meta = {
        "seed": seed, "mode": mode, "train_windows": int(len(tr)), "validation_windows": int(len(va)), "test_windows": int(len(te)),
        "gru": gru_info, "device_selected": str(next(gru.parameters()).device), "branch_reliability_temporal": rel_t, "branch_reliability_domain": rel_d,
        "platt_temporal": cal_t.to_dict(), "platt_domain": cal_d.to_dict(), "thresholds_validation_only": thresholds,
        "test_labels_used_for_threshold_selection": False,
        "svm_subsampled_training": bool(not smoke and seedspec.get("svm", {}).get("max_train_samples", None) is not None),
        "transformer_baseline_skipped": bool(not smoke and (seedspec.get("transformer", {}).get("skip", False) or seedspec.get("skip_optional_baselines", False))),
        "optional_baselines_skipped": bool(not smoke and seedspec.get("skip_optional_baselines", False)),
        "experiment_config": cfg if cfg is not None else {},
        "inference_timing_seconds": {"gru_attention_test_total": float(gru_test_seconds),
                                     "gru_attention_ms_per_window": float(1000*gru_test_seconds/max(len(xte),1)),
                                     "xgboost_test_total": float(xgb_test_seconds),
                                     "xgboost_ms_per_window": float(1000*xgb_test_seconds/max(len(dte),1)),
                                     "fusion_test_total": float(fusion_test_seconds),
                                     "fusion_ms_per_window": float(1000*fusion_test_seconds/max(len(yte),1))},
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return metric_rows, predictions, run_meta, {"xgb":xgb, "gru":gru, "p_test_map":p_test_map, "thresholds":thresholds,
                                               "p_t_test":p_t_test, "p_d_test":p_d_test, "p_f_test":p_f_test,
                                               "p_t_val":p_t_val, "p_d_val":p_d_val, "p_f_val":p_f_val,
                                               "y_test":yte, "y_val":yva, "xte":xte, "dte":dte,
                                               "wt_test":wt_test, "wd_test":wd_test, "attention":att_test,
                                               "domain_names":json.loads((Path("data/processed")/"domain_feature_columns.json").read_text()) if Path("data/processed/domain_feature_columns.json").exists() else [f"feature_{i}" for i in range(dte.shape[1])]}


def summarize_results(metric_rows, out_dir: Path):
    df = pd.DataFrame(metric_rows)
    df.to_csv(out_dir / "metrics_by_seed.csv", index=False, float_format="%.8f")
    summary=[]
    for (model, metric), group in df.melt(id_vars=["seed","model"], value_vars=["accuracy","precision","recall","f1","roc_auc","pr_auc","ece","brier","fpr","false_alarms_per_simulated_hour"]).groupby(["model","variable"]):
        vals = group["value"].to_numpy(); stats = mean_sd_ci(vals)
        summary.append({"model":model,"metric":metric,**stats})
    pd.DataFrame(summary).to_csv(out_dir / "metrics_summary.csv", index=False, float_format="%.8f")
    ablation_models = [x for x in ["Generic-feature XGBoost", "Domain-aware XGBoost", "Temporal branch only", "Fixed-weight fusion", "CyberRAX-CP"] if x in set(df.model)]
    if ablation_models:
        ab = df[df.model.isin(ablation_models)].copy()
        ab.to_csv(out_dir / "ablation_metrics_by_seed.csv", index=False, float_format="%.8f")
        ab.groupby("model")[["f1","roc_auc","ece","brier","fpr","false_alarms_per_simulated_hour"]].agg(["mean","std"]).to_csv(out_dir / "ablation_summary.csv", float_format="%.8f")
    if "CyberRAX-CP" in set(df.model):
        tests=[]
        for metric in ["f1", "ece", "brier"]:
            test = paired_wilcoxon_table(df, proposed="CyberRAX-CP", metric=metric)
            if not test.empty: tests.append(test)
        if tests: pd.concat(tests, ignore_index=True).to_csv(out_dir / "paired_wilcoxon_holm.csv", index=False, float_format="%.8f")
    return df


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke","quick","full"], default="smoke")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="results/smoke")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args=ap.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir=Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data=load_data(Path(args.data_dir))
    seeds=args.seeds if args.seeds else (cfg["smoke"]["seeds"] if args.mode=="smoke" else cfg["experiment"]["seeds"])
    all_metrics=[]; all_preds=[]; manifests=[]
    t0=time.time()
    for seed in seeds:
        print(f"[{args.mode}] seed={seed}", flush=True)
        rows, preds, meta, _ = run_one_seed(seed, data, out_dir, args.mode, cfg)
        all_metrics.extend(rows); all_preds.append(preds); manifests.append(meta)
        print(f"  completed seed {seed}: {len(rows)} evaluated configurations", flush=True)
    metrics_df=summarize_results(all_metrics, out_dir)
    pd.concat(all_preds, ignore_index=True).to_csv(out_dir / "predictions_test_all_seeds.csv", index=False, float_format="%.8f")
    (out_dir / "experiment_manifest.json").write_text(json.dumps({
        "project":"CyberRAX-CP", "mode":args.mode, "seeds":list(map(int,seeds)),
        "runtime_seconds":round(time.time()-t0,2), "python":platform.python_version(),
        "platform":platform.platform(), "torch":torch.__version__, "numpy":np.__version__,
        "rows_in_metrics":len(metrics_df), "provenance":"Reconstructed dataset; see top-level README.",
        "runs":manifests,
    }, indent=2), encoding="utf-8")
    print(metrics_df.groupby("model")["f1"].agg(["mean","std"]).to_string())
    print(f"Wrote results to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
