"""Proxy-label benchmark helpers (no ground-truth anomaly labels exist)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from . import config as C


def build_pool(all_df: pd.DataFrame, train_types) -> pd.DataFrame:
    """2024 records that can be scored, including data-quality anomalies (they are proxy positives)."""
    d = all_df[(all_df.year == C.TEST_YEAR) & all_df.status.isin(["model_ready", "dq_anomaly"])]
    d = d[(d.site_eui > 0) & d.gfa.notna() & d.prop_type.isin(set(train_types))]
    return d.reset_index(drop=True)


def proxy_labels(pool: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=pool.index)
    out["plausibility_fail"] = (pool.status == "dq_anomaly").to_numpy()
    out["yoy_shift_gt3x"] = ((pool.eui_yoy_ratio > 3) | (pool.eui_yoy_ratio < 1 / 3)).fillna(False).to_numpy()
    out["extreme_es_score"] = ((pool.es_score <= 5) | (pool.es_score >= 95)).fillna(False).to_numpy()
    out["any_proxy"] = out.any(axis=1)
    out["any_excl_plausibility"] = out[["yoy_shift_gt3x", "extreme_es_score"]].any(axis=1)
    return out


def precision_recall_at_k(y, score, frac):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-score)[:k]
    tp = y[top].sum()
    return float(tp / k), float(tp / max(1, y.sum()))


def benchmark(y, scores: dict, fracs=(0.05, 0.10)):
    rows = []
    for name, s in scores.items():
        s = np.asarray(s, float)
        s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1.0)   # unscorable rows rank last
        row = {"method": name, "pr_auc": float(average_precision_score(y, s)), "base_rate": float(np.mean(y))}
        for f in fracs:
            p, r = precision_recall_at_k(y, s, f)
            row[f"precision@{int(f*100)}%"], row[f"recall@{int(f*100)}%"] = p, r
        rows.append(row)
    return pd.DataFrame(rows)


def topk_jaccard(a, b, frac=0.10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = max(1, int(round(frac * len(a))))
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(ta & tb) / len(ta | tb)


def pairwise_stability(score_list, frac=0.10):
    sp, jc = [], []
    for i in range(len(score_list)):
        for j in range(i + 1, len(score_list)):
            sp.append(spearmanr(score_list[i], score_list[j]).statistic)
            jc.append(topk_jaccard(score_list[i], score_list[j], frac))
    return {"mean_spearman": float(np.mean(sp)), "min_spearman": float(np.min(sp)),
            "mean_top10_jaccard": float(np.mean(jc)), "min_top10_jaccard": float(np.min(jc))}
