from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulate import generate_dataset, ATTACK_TYPES
from preprocess import split_ranges, _impute_by_split, build_features, materialize_windows
from fusion import adaptive_fusion, expected_calibration_error


def test_generator_is_deterministic_and_has_seven_attack_types():
    a = generate_dataset(seed=123, n=600, train_end=360, validation_end=480)
    b = generate_dataset(seed=123, n=600, train_end=360, validation_end=480)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 600
    assert set(a.loc[a.label == 1, "attack_type"].unique()).issubset(set(ATTACK_TYPES))
    assert len(ATTACK_TYPES) == 7
    assert a.label.mean() > 0.3


def test_domain_feature_count_and_split_safe_windows(tmp_path):
    df = generate_dataset(seed=321, n=600, train_end=360, validation_end=480)
    ranges = split_ranges(len(df), 360, 480)
    imputed, flags, _ = _impute_by_split(df, ranges)
    feature_table, names, _, _ = build_features(df, imputed, flags, ranges, window=30)
    assert len(names) == 69
    assert feature_table.shape[0] == 600
    result = materialize_windows(df, imputed, feature_table, ranges, 30, tmp_path)
    idx = pd.read_csv(tmp_path / "window_index.csv")
    assert idx.groupby("split").size().to_dict() == {"test": 91, "train": 331, "validation": 91}
    for split, (a,b) in ranges.items():
        part = idx[idx.split == split]
        assert (part.global_start_index >= a).all()
        assert (part.global_end_index < b).all()
    assert result["domain_features_shape"][1] == 69


def test_adaptive_fusion_weights_normalize():
    p_t = np.array([0.1, 0.5, 0.9, 0.99])
    p_d = np.array([0.2, 0.4, 0.7, 0.98])
    p_f, w_t, w_d, c_t, c_d = adaptive_fusion(p_t, p_d, 25.0, 40.0)
    assert np.allclose(w_t + w_d, 1.0)
    assert np.all((p_f >= 0) & (p_f <= 1))
    assert np.all((c_t >= 0) & (c_t <= 1))
    assert np.all((c_d >= 0) & (c_d <= 1))


def test_ece_is_bounded():
    ece = expected_calibration_error(np.array([0,1,0,1]), np.array([.1,.9,.2,.8]), 5)
    assert 0 <= ece <= 1
