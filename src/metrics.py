"""Classification, calibration, confidence-interval and paired-test utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from fusion import expected_calibration_error


def classification_metrics(y_true, p, threshold: float = 0.5, n_bins: int = 15):
    y = np.asarray(y_true, dtype=int); p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "n": int(len(y)), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "ece": float(expected_calibration_error(y, p, n_bins)),
        "brier": float(brier_score_loss(y, p)),
        "fpr": float(fp/max(fp+tn, 1)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def mean_sd_ci(values, confidence: float = 0.95):
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    if len(x) == 0: return {"mean": float("nan"), "sd": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(x.mean()); sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    if len(x) > 1:
        crit = float(student_t.ppf((1+confidence)/2, len(x)-1)); half = crit*sd/np.sqrt(len(x))
    else: half = 0.0
    return {"mean": mean, "sd": sd, "ci_low": mean-half, "ci_high": mean+half, "n": int(len(x))}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float); n = len(p)
    order = np.argsort(p); adjusted = np.empty(n, dtype=float); running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n-rank)*p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def paired_wilcoxon_table(metrics_by_seed: pd.DataFrame, proposed: str = "CyberRAX-CP", metric: str = "f1"):
    rows = []
    if metrics_by_seed.empty: return pd.DataFrame(columns=["comparison", "metric", "statistic", "p_value", "p_holm"])
    piv = metrics_by_seed.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
    comps = [c for c in piv.columns if c != proposed]
    pvals = []
    for c in comps:
        a, b = piv[proposed].dropna().align(piv[c].dropna(), join="inner")
        if len(a) < 2 or np.allclose(a.values, b.values):
            stat, pv = float("nan"), 1.0
        else:
            res = wilcoxon(a.values, b.values, alternative="two-sided", zero_method="wilcox", method="auto")
            stat, pv = float(res.statistic), float(res.pvalue)
        rows.append({"comparison": f"{proposed} vs {c}", "metric": metric, "n_pairs": len(a), "statistic": stat, "p_value": pv})
        pvals.append(pv)
    if rows:
        adj = holm_adjust(pvals)
        for r, pa in zip(rows, adj): r["p_holm"] = float(pa)
    return pd.DataFrame(rows)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b)/len(a | b) if (a | b) else 1.0


def explanation_stability(top_feature_lists, rank_vectors):
    """Summarize all pairwise seed comparisons; inputs are lists ordered by importance."""
    jac, corr = [], []
    for i in range(len(top_feature_lists)):
        for j in range(i+1, len(top_feature_lists)):
            jac.append(jaccard(top_feature_lists[i][:5], top_feature_lists[j][:5]))
            a, b = np.asarray(rank_vectors[i], float), np.asarray(rank_vectors[j], float)
            rho = spearmanr(a, b).statistic
            corr.append(float(rho) if np.isfinite(rho) else 0.0)
    return {"top5_jaccard_mean": float(np.mean(jac)) if jac else float("nan"),
            "top5_jaccard_sd": float(np.std(jac, ddof=1)) if len(jac)>1 else 0.0,
            "rank_correlation_mean": float(np.mean(corr)) if corr else float("nan"),
            "rank_correlation_sd": float(np.std(corr, ddof=1)) if len(corr)>1 else 0.0,
            "pair_count": len(jac)}
