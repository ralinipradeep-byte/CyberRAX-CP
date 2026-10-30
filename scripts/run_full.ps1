$ErrorActionPreference = "Stop"
python -m pip install -r requirements.txt
python src/simulate.py --out data/raw/gpotsim_reconstructed.csv --seed 20260814
python src/preprocess.py --input data/raw/gpotsim_reconstructed.csv --out data/processed --window 30
pytest -q
python src/train.py --mode full --data-dir data/processed --out results/full
python src/robustness.py --data-dir data/processed --results-dir results/full
python src/explanation_stability.py --results-dir results/full
python src/make_figures.py --data data/raw/gpotsim_reconstructed.csv --results results/full --out figures/full
python src/hash_artifacts.py --root . --out data/metadata/SHA256SUMS.txt
