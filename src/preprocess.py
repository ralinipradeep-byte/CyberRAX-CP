#!/usr/bin/env python3
"""Leakage-safe preprocessing and materialization of the 69-feature and 30-step inputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from simulate import PROCESS_COLUMNS, NETWORK_COLUMNS

WINDOW = 30
ROLLING_SIGNALS = [
    "pressure_in", "pressure_mid", "pressure_out", "flow_main", "flow_branch",
    "valve_in", "valve_out", "temperature", "demand", "communication_delay_ms"
]


def split_ranges(n: int, train_end: int = 14400, validation_end: int = 19200):
    if not (0 < train_end < validation_end < n):
        raise ValueError("Invalid chronological split boundaries")
    return {"train": (0, train_end), "validation": (train_end, validation_end), "test": (validation_end, n)}


def _impute_by_split(df: pd.DataFrame, ranges: dict[str, tuple[int, int]]):
    numeric_cols = PROCESS_COLUMNS + NETWORK_COLUMNS
    imputed = df.copy()
    flags = pd.DataFrame(index=df.index)
    flags["missing_pressure_flag"] = df[["pressure_in", "pressure_mid", "pressure_out"]].isna().any(axis=1).astype(float)
    flags["missing_flow_flag"] = df[["flow_main", "flow_branch"]].isna().any(axis=1).astype(float)
    train_a, train_b = ranges["train"]
    train = df.iloc[train_a:train_b][numeric_cols].ffill()
    medians = train.median(numeric_only=True).fillna(0.0)
    for split, (a, b) in ranges.items():
        part = df.iloc[a:b][numeric_cols].copy().ffill().fillna(medians)
        imputed.loc[a:b-1, numeric_cols] = part.to_numpy()
    imputed[numeric_cols] = imputed[numeric_cols].astype(float)
    return imputed, flags, medians


def _rolling_features(part: pd.DataFrame, window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=part.index)
    for col in ROLLING_SIGNALS:
        roll = part[col].rolling(window=window, min_periods=1)
        out[f"{col}_rolling_mean"] = roll.mean()
        out[f"{col}_rolling_std"] = roll.std(ddof=0).fillna(0.0)
    return out


def build_features(df: pd.DataFrame, imputed: pd.DataFrame, flags: pd.DataFrame,
                   ranges: dict[str, tuple[int, int]], window: int = WINDOW):
    train_a, train_b = ranges["train"]
    train_normal = imputed.iloc[train_a:train_b]["label"].to_numpy() == 0
    tr = imputed.iloc[train_a:train_b].reset_index(drop=True)
    tr_normal = tr.loc[train_normal]
    pressure_predictors = ["pressure_in", "flow_main", "valve_in", "pressure_setpoint", "demand", "valve_out", "compressor_state"]
    flow_predictors = ["pressure_mid", "pressure_out", "flow_main", "valve_out", "demand", "compressor_state"]
    p_model = Ridge(alpha=1.0).fit(tr_normal[pressure_predictors], tr_normal["pressure_mid"])
    f_model = Ridge(alpha=1.0).fit(tr_normal[flow_predictors], tr_normal["flow_branch"])
    p_res_train = tr_normal["pressure_mid"].to_numpy() - p_model.predict(tr_normal[pressure_predictors])
    f_res_train = tr_normal["flow_branch"].to_numpy() - f_model.predict(tr_normal[flow_predictors])
    p_scale = max(float(np.std(p_res_train)), 1e-6)
    f_scale = max(float(np.std(f_res_train)), 1e-6)
    delay_mu = float(tr_normal["communication_delay_ms"].mean())
    delay_sd = max(float(tr_normal["communication_delay_ms"].std(ddof=0)), 1e-6)
    packet_mu = float(tr_normal["packet_rate"].mean())
    packet_sd = max(float(tr_normal["packet_rate"].std(ddof=0)), 1e-6)

    pieces = []
    for split, (a, b) in ranges.items():
        part = imputed.iloc[a:b].reset_index(drop=True)
        part_raw = df.iloc[a:b].reset_index(drop=True)
        part_flags = flags.iloc[a:b].reset_index(drop=True)
        raw15 = part[PROCESS_COLUMNS].copy()
        rolling = _rolling_features(part, window)
        network = part[NETWORK_COLUMNS].copy()
        p_res = part["pressure_mid"].to_numpy() - p_model.predict(part[pressure_predictors])
        f_res = part["flow_branch"].to_numpy() - f_model.predict(part[flow_predictors])
        command_mismatch = np.abs(part["control_command"].to_numpy() - part["valve_out"].to_numpy())
        command_transition = part["control_command"].diff().abs().rolling(window=window, min_periods=1).mean().fillna(0).to_numpy()
        fc_transition = part["modbus_function_code"].diff().fillna(0).ne(0).astype(float).to_numpy()
        device_change = part["device_id"].diff().fillna(0).ne(0).astype(float).to_numpy()
        engineered = pd.DataFrame({
            "pressure_residual": p_res,
            "flow_residual": f_res,
            "pressure_residual_norm": p_res / p_scale,
            "flow_residual_norm": f_res / f_scale,
            "command_response_mismatch": command_mismatch,
            "command_transition_rate": command_transition,
            "function_code_transition_rate": fc_transition,
            "device_change_flag": device_change,
            "delay_zscore": (part["communication_delay_ms"].to_numpy() - delay_mu) / delay_sd,
            "packet_rate_zscore": (part["packet_rate"].to_numpy() - packet_mu) / packet_sd,
            "missing_pressure_flag": part_flags["missing_pressure_flag"].to_numpy(),
            "missing_flow_flag": part_flags["missing_flow_flag"].to_numpy(),
            "high_load_mode": (part["mode"].to_numpy() == "high_load").astype(float),
            "maintenance_mode": (part["mode"].to_numpy() == "maintenance").astype(float),
        })
        feat = pd.concat([raw15.reset_index(drop=True), rolling.reset_index(drop=True), network.reset_index(drop=True), engineered], axis=1)
        if feat.shape[1] != 69:
            raise AssertionError(f"Expected 69 domain features, found {feat.shape[1]}")
        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feat.insert(0, "sample_id", part["sample_id"].to_numpy())
        feat.insert(1, "split", split)
        feat.insert(2, "mode", part["mode"].to_numpy())
        feat.insert(3, "attack_type", part["attack_type"].to_numpy())
        feat.insert(4, "label", part["label"].to_numpy(dtype=np.int8))
        pieces.append(feat)
    feature_table = pd.concat(pieces, ignore_index=True)
    feature_names = [c for c in feature_table.columns if c not in {"sample_id", "split", "mode", "attack_type", "label"}]
    if len(feature_names) != 69 or len(set(feature_names)) != 69:
        raise AssertionError("Domain feature names are not unique or do not total 69")
    return feature_table, feature_names, p_model, f_model


def materialize_windows(df: pd.DataFrame, imputed: pd.DataFrame, feature_table: pd.DataFrame,
                        ranges: dict[str, tuple[int, int]], window: int, out_dir: Path):
    temporal_scaler = StandardScaler().fit(imputed.iloc[ranges["train"][0]:ranges["train"][1]][PROCESS_COLUMNS].to_numpy())
    domain_names = [c for c in feature_table.columns if c not in {"sample_id", "split", "mode", "attack_type", "label"}]
    train_rows = feature_table[feature_table.split.eq("train")][domain_names]
    domain_scaler = StandardScaler().fit(train_rows.to_numpy())
    all_raw_windows, all_scaled_windows, all_domain_raw, all_domain_scaled, all_y, metadata = [], [], [], [], [], []
    for split, (a, b) in ranges.items():
        raw_x = imputed.iloc[a:b][PROCESS_COLUMNS].to_numpy(dtype=np.float64)
        scaled_x = temporal_scaler.transform(raw_x).astype(np.float64)
        part_labels = df.iloc[a:b]["label"].to_numpy(dtype=np.int8)
        part_attack = df.iloc[a:b]["attack_type"].astype(str).to_numpy()
        part_features = feature_table.iloc[a:b][domain_names].to_numpy(dtype=np.float64)
        if len(raw_x) < window:
            continue
        raw_w = np.lib.stride_tricks.sliding_window_view(raw_x, window_shape=window, axis=0).transpose(0, 2, 1).copy()
        scaled_w = np.lib.stride_tricks.sliding_window_view(scaled_x, window_shape=window, axis=0).transpose(0, 2, 1).copy()
        y = part_labels[window-1:]
        domain = part_features[window-1:]
        domain_scaled = domain_scaler.transform(domain).astype(np.float64)
        if raw_w.shape[1:] != (window, len(PROCESS_COLUMNS)):
            raise AssertionError(f"Unexpected window shape: {raw_w.shape}")
        all_raw_windows.append(raw_w)
        all_scaled_windows.append(scaled_w)
        all_domain_raw.append(domain)
        all_domain_scaled.append(domain_scaled)
        all_y.append(y)
        for local_end in range(window-1, len(raw_x)):
            global_end = a + local_end
            metadata.append({
                "window_id": len(metadata), "split": split,
                "global_start_index": global_end-window+1,
                "global_end_index": global_end,
                "start_sample_id": int(df.iloc[global_end-window+1]["sample_id"]),
                "end_sample_id": int(df.iloc[global_end]["sample_id"]),
                "t_seconds": int(df.iloc[global_end]["t_seconds"]),
                "label": int(part_labels[local_end]), "attack_type": part_attack[local_end],
            })
    raw_windows = np.concatenate(all_raw_windows).astype(np.float64, copy=False)
    scaled_windows = np.concatenate(all_scaled_windows).astype(np.float64, copy=False)
    domain_raw = np.concatenate(all_domain_raw).astype(np.float64, copy=False)
    domain_scaled = np.concatenate(all_domain_scaled).astype(np.float64, copy=False)
    y = np.concatenate(all_y).astype(np.int8, copy=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "windows_raw_float64.npy", raw_windows, allow_pickle=False)
    np.save(out_dir / "windows_standardized_float64.npy", scaled_windows, allow_pickle=False)
    np.save(out_dir / "domain_features_float64.npy", domain_raw, allow_pickle=False)
    np.save(out_dir / "domain_features_standardized_float64.npy", domain_scaled, allow_pickle=False)
    np.save(out_dir / "y_windows_int8.npy", y, allow_pickle=False)
    pd.DataFrame(metadata).to_csv(out_dir / "window_index.csv", index=False)
    feature_table.to_csv(out_dir / "feature_table_69.csv", index=False, float_format="%.8f")
    split_index = pd.DataFrame({"sample_id": df["sample_id"], "t_seconds": df["t_seconds"]})
    split_index["split"] = "test"
    split_index.loc[:ranges["train"][1]-1, "split"] = "train"
    split_index.loc[ranges["train"][1]:ranges["validation"][1]-1, "split"] = "validation"
    split_index.to_csv(out_dir / "split_indices.csv", index=False)
    temporal_scaler_path = out_dir / "temporal_scaler.joblib"
    domain_scaler_path = out_dir / "domain_scaler.joblib"
    joblib.dump(temporal_scaler, temporal_scaler_path)
    joblib.dump(domain_scaler, domain_scaler_path)
    return {
        "temporal_scaler": temporal_scaler, "domain_scaler": domain_scaler,
        "raw_windows_shape": list(raw_windows.shape), "scaled_windows_shape": list(scaled_windows.shape),
        "domain_features_shape": list(domain_raw.shape), "y_shape": list(y.shape),
        "split_window_counts": pd.DataFrame(metadata).groupby("split").size().astype(int).to_dict(),
        "temporal_input_columns": PROCESS_COLUMNS,
        "domain_feature_columns": [c for c in feature_table.columns if c not in {"sample_id", "split", "mode", "attack_type", "label"}],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/gpotsim_reconstructed.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--train-end", type=int, default=14400)
    ap.add_argument("--validation-end", type=int, default=19200)
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp)
    ranges = split_ranges(len(df), args.train_end, args.validation_end)
    imputed, flags, medians = _impute_by_split(df, ranges)
    feature_table, feature_names, p_model, f_model = build_features(df, imputed, flags, ranges, args.window)
    result = materialize_windows(df, imputed, feature_table, ranges, args.window, out)
    joblib.dump({"pressure_model": p_model, "flow_model": f_model, "imputation_medians": medians.to_dict()}, out / "physical_models_and_imputation.joblib")
    (out / "temporal_input_columns.json").write_text(json.dumps(PROCESS_COLUMNS, indent=2), encoding="utf-8")
    (out / "domain_feature_columns.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    metadata = {
        "provenance": "Preprocessed deterministic reconstruction; not the unavailable original raw dataset.",
        "input_csv": str(inp), "observations": len(df), "window_length": args.window,
        "chronological_ranges_zero_based_inclusive_exclusive": {k: list(v) for k, v in ranges.items()},
        "domain_feature_count": len(feature_names), "temporal_input_count": len(PROCESS_COLUMNS),
        "split_window_counts": result["split_window_counts"],
        "shapes": {k: v for k, v in result.items() if k.endswith("shape")},
        "standardization_scope": "fit on training observations only",
        "physical_residual_fit_scope": "training rows with label normal only",
        "window_policy": "windows generated independently within each split; no cross-split windows",
        "storage_dtype": "float64 for audit arrays; training casts to float32",
    }
    (out / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
