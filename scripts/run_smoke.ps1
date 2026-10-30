$ErrorActionPreference = "Stop"
python -m pip install -r requirements.txt
pytest -q
python src/train.py --mode smoke --data-dir data/processed --out results/smoke
python src/robustness.py --data-dir data/processed --results-dir results/smoke
python src/make_figures.py --data data/raw/gpotsim_reconstructed.csv --results results/smoke --out figures/smoke
