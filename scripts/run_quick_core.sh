#!/usr/bin/env bash
set -euo pipefail
python src/preprocess.py --input data/raw/gpotsim_reconstructed.csv --out data/processed
python -m pytest -q
python src/train.py --mode quick --config configs/ten_seed_quick_check.yaml --data-dir data/processed --out results/quick10_new
python src/robustness.py --data-dir data/processed --results-dir results/quick10_new
python src/explanation_stability.py --results-dir results/quick10_new --feature-names data/processed/domain_feature_columns.json
python src/make_figures.py --results results/quick10_new --out figures/quick10_new
python src/plot_audit.py --results results/quick10_new --out figures/quick10_new
