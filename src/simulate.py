#!/usr/bin/env python3
"""Deterministic reconstruction of the GP-OTSim-style SCADA testbed described in the manuscript.

This is an openly documented reconstruction, not a recovered copy of any unavailable original data.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ATTACK_TYPES = [
    "false_data_injection", "replay", "command_injection",
    "denial_of_service", "scanning", "man_in_the_middle", "device_spoofing"
]
MODES = ["normal", "high_load", "maintenance"]
PROCESS_COLUMNS = [
    "pressure_in", "pressure_mid", "pressure_out", "flow_main", "flow_branch",
    "valve_in", "valve_out", "compressor_state", "temperature", "demand",
    "pressure_setpoint", "control_command", "modbus_function_code", "device_id",
    "communication_delay_ms"
]
NETWORK_COLUMNS = [
    "packet_rate", "packet_size_bytes", "interarrival_jitter_ms", "retry_count",
    "tcp_syn_rate", "read_write_ratio", "response_time_ms", "request_count",
    "response_count", "error_response_rate", "address_diversity",
    "connection_reset_rate", "modbus_exception_rate", "command_burst_rate",
    "network_load", "response_missing_rate", "protocol_entropy", "source_ip_count",
    "destination_port_diversity", "session_duration_s"
]


def _attack_labels_for_split(n: int, target_count: int, rng: np.random.Generator) -> np.ndarray:
    """Create attack episodes while enforcing the documented split-level prevalence."""
    labels = np.full(n, "normal", dtype=object)
    occupied = np.zeros(n, dtype=bool)
    remaining = int(target_count)
    cursor = int(rng.integers(0, 30))
    event_idx = 0
    while remaining > 0 and cursor < n:
        cursor += int(rng.integers(4, 28))
        if cursor >= n:
            break
        length = min(int(rng.integers(18, 76)), remaining, n - cursor)
        # Avoid overwriting an earlier episode. The cursor advances past each segment.
        if length <= 0:
            break
        attack = ATTACK_TYPES[event_idx % len(ATTACK_TYPES)]
        labels[cursor:cursor + length] = attack
        occupied[cursor:cursor + length] = True
        remaining -= length
        cursor += length
        event_idx += 1
    if remaining > 0:
        available = np.flatnonzero(~occupied)
        chosen = rng.choice(available, size=min(remaining, len(available)), replace=False)
        labels[chosen] = rng.choice(ATTACK_TYPES, size=len(chosen), replace=True)
    # Randomly permute class identities across the occupied positions without changing the attack mask.
    attack_idx = np.flatnonzero(labels != "normal")
    if len(attack_idx):
        # Assign class labels in blocks to retain local episodes, then lightly randomize block types.
        for j, idx in enumerate(attack_idx):
            if rng.random() < 0.035:
                labels[idx] = rng.choice(ATTACK_TYPES)
    return labels


def generate_dataset(seed: int = 20260814, n: int = 24000,
                     train_end: int = 14400, validation_end: int = 19200) -> pd.DataFrame:
    if n != 24000 or train_end != 14400 or validation_end != 19200:
        # The generator supports alternate counts for tests, but documented shares are applied per split.
        if not (0 < train_end < validation_end < n):
            raise ValueError("Require 0 < train_end < validation_end < n")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64)
    mode_idx = (t // max(1, n // 6)) % 3
    mode = np.array(MODES, dtype=object)[mode_idx]
    high = (mode == "high_load").astype(float)
    maint = (mode == "maintenance").astype(float)

    demand = 1.0 + 0.13*np.sin(t/290) + 0.07*np.sin(t/37) + 0.34*high - 0.22*maint + rng.normal(0, .035, n)
    valve_in = np.clip(.72 + .10*np.sin(t/210) + .09*high - .18*maint + rng.normal(0, .025, n), .12, 1.0)
    valve_out = np.clip(.66 + .12*np.sin(t/180 + .7) + .07*high - .16*maint + rng.normal(0, .03, n), .10, 1.0)
    pressure_in = 66.0 + 1.25*np.sin(t/330) + .45*np.sin(t/77) + .7*high - 1.0*maint + rng.normal(0, .22, n)
    flow_main = np.maximum(.03, demand*valve_in*1.13 + rng.normal(0, .045, n))
    pressure_mid = pressure_in - 3.7*flow_main/(.45 + valve_in) + rng.normal(0, .18, n)
    flow_branch = np.maximum(.02, demand*valve_out*.79 + rng.normal(0, .035, n))
    pressure_out = pressure_mid - 4.7*flow_branch/(.42 + valve_out) + rng.normal(0, .20, n)
    compressor_state = ((np.sin(t/950) > -.2) & (maint < .5)).astype(float)
    temperature = 25.0 + 1.8*np.sin(t/620) + .7*high - .5*maint + rng.normal(0, .25, n)
    pressure_setpoint = 61.0 + .65*high - .85*maint + .35*np.sin(t/500)
    control_command = np.clip(valve_out + rng.normal(0, .035, n), 0, 1)
    modbus_function_code = rng.choice([3, 4, 5, 6, 16], size=n, p=[.48, .18, .10, .16, .08]).astype(float)
    device_id = rng.choice([1, 2, 3, 4], size=n, p=[.34, .29, .23, .14]).astype(float)
    communication_delay_ms = np.clip(10.5 + np.abs(rng.normal(0, 2.4, n)) + 1.0*high + 2.0*maint, 2, None)

    # The manuscript reports 41.5%, 39.8%, and 40.7% attack shares in the three chronological partitions.
    boundaries = [(0, train_end, .415), (train_end, validation_end, .398), (validation_end, n, .407)]
    attack_type = np.full(n, "normal", dtype=object)
    for a, b, share in boundaries:
        target = int(round((b-a)*share))
        attack_type[a:b] = _attack_labels_for_split(b-a, target, rng)
    label = (attack_type != "normal").astype(np.int8)

    # Generate network telemetry first; attacks alter process and communication observations below.
    packet_rate = np.clip(85 + 12*np.sin(t/53) + 8*high - 12*maint + rng.normal(0, 7, n), 1, None)
    packet_size_bytes = np.clip(rng.normal(170, 25, n) + 8*high, 40, 1500)
    interarrival_jitter_ms = np.clip(np.abs(rng.normal(2.5, .9, n)) + .2*maint, 0, None)
    retry_count = rng.poisson(.12, n).astype(float)
    tcp_syn_rate = np.clip(rng.normal(1.2, .3, n), 0, None)
    read_write_ratio = np.clip(rng.normal(4.0, .7, n), .1, None)
    response_time_ms = communication_delay_ms + np.abs(rng.normal(4, 1.5, n))
    request_count = packet_rate * rng.uniform(.45, .60, n)
    response_count = np.maximum(0, request_count - rng.poisson(1.0, n))
    error_response_rate = np.clip(rng.beta(1.2, 30, n), 0, 1)
    address_diversity = np.clip(rng.normal(3.5, .8, n), 1, 32)
    connection_reset_rate = np.clip(rng.beta(1, 40, n), 0, 1)
    modbus_exception_rate = np.clip(rng.beta(1, 45, n), 0, 1)
    command_burst_rate = np.clip(rng.normal(.15, .06, n), 0, None)
    network_load = np.clip(packet_rate*packet_size_bytes/100000, 0, None)
    response_missing_rate = np.clip(rng.beta(1, 60, n), 0, 1)
    protocol_entropy = np.clip(rng.normal(1.15, .18, n), 0, 4)
    source_ip_count = np.clip(rng.poisson(2.0, n)+1, 1, 100).astype(float)
    destination_port_diversity = np.clip(rng.poisson(2.2, n)+1, 1, 100).astype(float)
    session_duration_s = np.clip(rng.normal(60, 15, n), 1, None)

    # Apply class-specific effects. No label/attack_type column is used as a model feature.
    for attack in ATTACK_TYPES:
        ix = np.flatnonzero(attack_type == attack)
        if not len(ix):
            continue
        if attack == "false_data_injection":
            sign = rng.choice([-1.0, 1.0], len(ix))
            pressure_mid[ix] += sign*rng.uniform(1.8, 5.5, len(ix))
            pressure_out[ix] += sign*rng.uniform(1.2, 4.2, len(ix))
            flow_main[ix] *= rng.uniform(.62, 1.42, len(ix))
            flow_branch[ix] *= rng.uniform(.65, 1.35, len(ix))
        elif attack == "replay":
            lag = 40
            for col in (pressure_in, pressure_mid, pressure_out, flow_main, flow_branch, valve_in, valve_out):
                col[ix] = col[np.maximum(ix-lag, 0)]
            communication_delay_ms[ix] += rng.uniform(1, 7, len(ix))
        elif attack == "command_injection":
            control_command[ix] = np.clip(control_command[ix] + rng.choice([-1, 1], len(ix))*rng.uniform(.25, .65, len(ix)), 0, 1)
            valve_out[ix] = np.clip(control_command[ix] + rng.normal(0, .08, len(ix)), 0, 1)
            compressor_state[ix] = 1 - compressor_state[ix]
            command_burst_rate[ix] += rng.uniform(.5, 1.4, len(ix))
        elif attack == "denial_of_service":
            communication_delay_ms[ix] *= rng.uniform(2.5, 6.0, len(ix))
            packet_rate[ix] *= rng.uniform(.12, .48, len(ix))
            retry_count[ix] += rng.poisson(3.0, len(ix))
            response_missing_rate[ix] = np.clip(response_missing_rate[ix] + rng.uniform(.25, .7, len(ix)), 0, 1)
            response_count[ix] *= rng.uniform(.1, .5, len(ix))
        elif attack == "scanning":
            packet_rate[ix] *= rng.uniform(1.8, 4.2, len(ix))
            source_ip_count[ix] += rng.integers(3, 14, len(ix))
            destination_port_diversity[ix] += rng.integers(4, 18, len(ix))
            tcp_syn_rate[ix] *= rng.uniform(2, 6, len(ix))
            modbus_function_code[ix] = rng.choice([1, 2, 8, 43], len(ix))
        elif attack == "man_in_the_middle":
            communication_delay_ms[ix] += rng.uniform(12, 55, len(ix))
            response_time_ms[ix] += rng.uniform(10, 50, len(ix))
            protocol_entropy[ix] += rng.uniform(.3, 1.1, len(ix))
            control_command[ix] = np.clip(control_command[ix] + rng.normal(.10, .15, len(ix)), 0, 1)
        elif attack == "device_spoofing":
            device_id[ix] = rng.choice([5, 6, 7, 8], len(ix))
            modbus_function_code[ix] = rng.choice([3, 4, 5, 6, 16, 43], len(ix))
            address_diversity[ix] += rng.uniform(2, 9, len(ix))
            pressure_in[ix] += rng.normal(1.2, 2.2, len(ix))

    # Recompute a few dependent network measures after attack effects.
    response_time_ms = communication_delay_ms + np.abs(response_time_ms - communication_delay_ms)
    network_load = np.clip(packet_rate*packet_size_bytes/100000, 0, None)
    read_write_ratio = np.clip(read_write_ratio + .2*(modbus_function_code >= 5), .1, None)
    error_response_rate = np.clip(error_response_rate + .08*(attack_type == "denial_of_service") + .03*(attack_type == "scanning"), 0, 1)
    protocol_entropy = np.clip(protocol_entropy, 0, 4)

    df = pd.DataFrame({
        "sample_id": np.arange(n, dtype=np.int32), "t_seconds": t,
        "mode": mode, "attack_type": attack_type, "label": label,
        "pressure_in": pressure_in, "pressure_mid": pressure_mid, "pressure_out": pressure_out,
        "flow_main": flow_main, "flow_branch": flow_branch, "valve_in": valve_in,
        "valve_out": valve_out, "compressor_state": compressor_state, "temperature": temperature,
        "demand": demand, "pressure_setpoint": pressure_setpoint, "control_command": control_command,
        "modbus_function_code": modbus_function_code, "device_id": device_id,
        "communication_delay_ms": communication_delay_ms,
        "packet_rate": packet_rate, "packet_size_bytes": packet_size_bytes,
        "interarrival_jitter_ms": interarrival_jitter_ms, "retry_count": retry_count,
        "tcp_syn_rate": tcp_syn_rate, "read_write_ratio": read_write_ratio,
        "response_time_ms": response_time_ms, "request_count": request_count,
        "response_count": response_count, "error_response_rate": error_response_rate,
        "address_diversity": address_diversity, "connection_reset_rate": connection_reset_rate,
        "modbus_exception_rate": modbus_exception_rate, "command_burst_rate": command_burst_rate,
        "network_load": network_load, "response_missing_rate": response_missing_rate,
        "protocol_entropy": protocol_entropy, "source_ip_count": source_ip_count,
        "destination_port_diversity": destination_port_diversity, "session_duration_s": session_duration_s,
    })

    # Natural telemetry loss is preserved as NaN in the raw CSV; flags are generated before imputation.
    miss_p = rng.random(n) < .004
    miss_f = rng.random(n) < .004
    for c in ["pressure_in", "pressure_mid", "pressure_out"]:
        df.loc[miss_p, c] = np.nan
    for c in ["flow_main", "flow_branch"]:
        df.loc[miss_f, c] = np.nan
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/gpotsim_reconstructed.csv")
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--train-end", type=int, default=14400)
    ap.add_argument("--validation-end", type=int, default=19200)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.seed, args.n, args.train_end, args.validation_end)
    df.to_csv(out, index=False, float_format="%.8f")
    train_end, val_end = args.train_end, args.validation_end
    metadata = {
        "dataset_name": "GP-OTSim-style deterministic reconstruction",
        "provenance": "Generated from manuscript method description; not the unavailable original raw dataset.",
        "generator_seed": args.seed, "observations": len(df), "sampling_interval_seconds": 1,
        "split_ranges_zero_based_inclusive_exclusive": {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, len(df)]},
        "mode_counts": df["mode"].value_counts().to_dict(),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "attack_type_counts": {str(k): int(v) for k, v in df["attack_type"].value_counts().to_dict().items()},
        "attack_types": ATTACK_TYPES,
        "process_input_columns": PROCESS_COLUMNS,
        "network_input_columns": NETWORK_COLUMNS,
        "missing_value_counts": {str(k): int(v) for k, v in df.isna().sum().to_dict().items() if v},
    }
    meta_path = out.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(df):,} rows to {out}")
    print("Attack share:", round(float(df.label.mean()), 4))
    print("Attack types:", df.loc[df.label.eq(1), "attack_type"].value_counts().to_dict())

if __name__ == "__main__":
    main()
