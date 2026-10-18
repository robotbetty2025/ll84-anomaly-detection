"""Step 2: rule-based baseline, reference regressors, and LightGBM residual model with grouped CV."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from src import config as C
from src.baseline import PercentileBaseline
from src.features import CAT_FEATURES, NUM_FEATURES, TARGET, sparse_features
from src.models import make_pipeline, standardized_residual

t0 = time.time()
sns.set_theme(style="whitegrid")
df = pd.read_parquet(C.PROCESSED / "ll84_model_ready.parquet")
train = df[df.year.isin(C.TRAIN_YEARS)].reset_index(drop=True)
test = df[df.year == C.TEST_YEAR].reset_index(drop=True)
dropped = sparse_features(train, NUM_FEATURES)
num_features = [c for c in NUM_FEATURES if c not in dropped]
feats = CAT_FEATURES + num_features
print("train", len(train), "test", len(test), "| dropped sparse:", dropped, flush=True)
test_seen = test[test.prop_type.isin(set(train.prop_type))].reset_index(drop=True)
metrics = {"n_train": len(train), "n_test": len(test), "n_test_seen_types": len(test_seen),
           "train_buildings": int(train.property_id.nunique()),
           "test_buildings_also_in_train_pct": float(test.drop_duplicates("property_id").property_id.isin(train.property_id).mean()),
           "dropped_sparse_features": dropped, "features_used": feats}
test = test_seen


def reg_metrics(y_log, p_log):
    y, p = np.exp(y_log), np.exp(p_log)
    return {"rmse_log": float(np.sqrt(mean_squared_error(y_log, p_log))), "r2_log": float(r2_score(y_log, p_log)),
            "rmse_eui": float(np.sqrt(mean_squared_error(y, p))), "mape_eui": float(np.mean(np.abs(y - p) / y))}


# ---- 1. rule-based baseline (P90 within property type, fit on train years)
base = PercentileBaseline(0.90).fit(train)
test["baseline_ratio"] = base.score(test)
test["baseline_flag"] = base.flag(test)
metrics["baseline_flag_rate_train"] = float(base.flag(train).mean())
metrics["baseline_flag_rate_test"] = float(test.baseline_flag.mean())

# ---- 2. reference regressors for the evaluation layer (property-type median, linear regression)
type_med = train.groupby("prop_type")[TARGET].median()
p_med = test.prop_type.map(type_med)
metrics["ref_type_median"] = reg_metrics(test[TARGET], p_med)
Xtr = pd.get_dummies(train[CAT_FEATURES].fillna("NA"), dtype=float)
Xte = pd.get_dummies(test[CAT_FEATURES].fillna("NA"), dtype=float).reindex(columns=Xtr.columns, fill_value=0)
med = train[num_features].median()
Xtr = pd.concat([Xtr, train[num_features].fillna(med)], axis=1)
Xte = pd.concat([Xte, test[num_features].fillna(med)], axis=1)
lin = LinearRegression().fit(Xtr, train[TARGET])
metrics["ref_linear_regression"] = reg_metrics(test[TARGET], lin.predict(Xte))

# ---- 3. LightGBM with 5-fold CV grouped by Property ID
pipe = make_pipeline(num_features)
space = {
    "model__n_estimators": randint(150, 600), "model__learning_rate": uniform(0.02, 0.13),
    "model__max_depth": randint(4, 12), "model__num_leaves": randint(15, 120),
    "model__subsample": uniform(0.6, 0.4), "model__subsample_freq": [1],
    "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_samples": randint(10, 80),
}
cv = GroupKFold(n_splits=5)
# Randomized search, resumable: each candidate (5-fold grouped CV) is cached so long searches can span several runs.
from sklearn.model_selection import ParameterSampler, cross_val_score
N_ITER, BUDGET_S = 40, 140
cache_path = C.REPORTS / "cv_search_cache.json"
import hashlib
fingerprint = hashlib.md5(json.dumps([feats, len(train), C.RANDOM_STATE, N_ITER, sorted(map(str, space))]).encode()).hexdigest()
raw_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
cache = raw_cache.get("results", {}) if raw_cache.get("fingerprint") == fingerprint else {}   # stale cache is ignored
cands = list(ParameterSampler(space, n_iter=N_ITER, random_state=C.RANDOM_STATE))
for i, p in enumerate(cands):
    if str(i) in cache:
        continue
    if time.time() - t0 > BUDGET_S:
        print(f"PARTIAL: {len(cache)}/{N_ITER} candidates done; re-run to continue", flush=True); sys.exit(0)
    sc = cross_val_score(pipe.set_params(**p), train[feats], train[TARGET], groups=train.property_id, cv=cv, n_jobs=2,
                         scoring="neg_root_mean_squared_error", params={"model__categorical_feature": [0, 1]})
    cache[str(i)] = {"params": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in p.items()}, "cv_rmse_log": float(-sc.mean())}
    cache_path.write_text(json.dumps({"fingerprint": fingerprint, "results": cache}, indent=1)); print(i, round(-sc.mean(), 4), round(time.time() - t0), "s", flush=True)
best_i = min(cache, key=lambda k: cache[k]["cv_rmse_log"])
best_params = cands[int(best_i)]
best = pipe.set_params(**best_params).fit(train[feats], train[TARGET], model__categorical_feature=[0, 1])
metrics["lgbm_best_params"] = cache[best_i]["params"]
metrics["lgbm_cv_rmse_log"] = cache[best_i]["cv_rmse_log"]
metrics["lgbm_search_candidates"] = {k: v["cv_rmse_log"] for k, v in cache.items()}
print("search done", round(time.time() - t0), "s | best cv rmse(log)", metrics["lgbm_cv_rmse_log"], flush=True)

# out-of-fold residuals on train -> sigma used to standardize residuals
oof = cross_val_predict(best, train[feats], train[TARGET], groups=train.property_id, cv=cv, n_jobs=2,
                        params={"model__categorical_feature": [0, 1]})
sigma = float(np.std(train[TARGET] - oof))
metrics["residual_sigma_log"] = sigma
metrics["lgbm_oof_train"] = reg_metrics(train[TARGET], oof)
train.assign(oof_log_pred=oof)[["property_id", "year", "prop_type", TARGET, "oof_log_pred"]].to_parquet(C.PROCESSED / "train_oof.parquet", index=False)

# ---- 4. evaluate on the held-out 2024 data
pred = best.predict(test[feats])
test["pred_log_eui"] = pred
test["pred_eui"] = np.exp(pred)
test["std_resid"] = standardized_residual(test[TARGET], pred, sigma)
metrics["lgbm_test_2024"] = reg_metrics(test[TARGET], pred)
thr = test.std_resid.quantile(0.90)            # flag the top 10%, same flag rate as the baseline
test["lgbm_flag"] = test.std_resid >= thr
a, b = test.baseline_flag, test.lgbm_flag
metrics["flag_agreement"] = {"baseline_flags": int(a.sum()), "lgbm_flags": int(b.sum()), "both": int((a & b).sum()),
                             "jaccard": float((a & b).sum() / (a | b).sum()),
                             "spearman_baseline_vs_lgbm_rank": float(test[["baseline_ratio", "std_resid"]].corr(method="spearman").iloc[0, 1])}
metrics["pct_abs_std_resid_gt_3"] = float((test.std_resid.abs() > 3).mean())
# residual symmetry audit across the biggest classes
big = test.prop_type.value_counts().head(10).index
metrics["residual_by_type_2024"] = test[test.prop_type.isin(big)].groupby("prop_type").std_resid.agg(["count", "mean", "std"]).round(3).to_dict("index")
metrics["flag_rate_by_type_2024"] = {
    t: {"baseline": round(float(g.baseline_flag.mean()), 3), "lgbm": round(float(g.lgbm_flag.mean()), 3), "n": len(g)}
    for t, g in test[test.prop_type.isin(big)].groupby("prop_type")}
imp = pd.Series(best.named_steps["model"].booster_.feature_importance("gain"), index=best.named_steps["model"].booster_.feature_name())
metrics["feature_importance_gain_pct"] = (imp / imp.sum() * 100).round(2).sort_values(ascending=False).to_dict()

# ---- 5. save artifacts
C.MODELS.mkdir(exist_ok=True)
joblib.dump({"pipeline": best, "features": feats, "sigma": sigma, "flag_threshold": float(thr)}, C.MODELS / "lightgbm_residual.joblib")
cols = ["property_id", "property_name", "address", "borough", "prop_type", "year_built", "gfa", "site_eui", "pred_eui",
        "std_resid", "baseline_ratio", "baseline_flag", "lgbm_flag"]
test[cols].to_parquet(C.PROCESSED / "scored_2024.parquet", index=False)
(C.REPORTS / "week4_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].hexbin(np.exp(test[TARGET]), test.pred_eui, gridsize=70, bins="log", xscale="log", yscale="log", cmap="viridis")
lim = [10, 600]; axes[0].plot(lim, lim, "r--", lw=1); axes[0].set_xlabel("actual site EUI"); axes[0].set_ylabel("expected EUI (LightGBM)")
axes[0].set_title("2024 hold-out: actual vs expected EUI")
sns.histplot(test.std_resid, bins=100, ax=axes[1], color="#2a7f62"); axes[1].axvline(thr, color="r", ls="--", label="top-10% cutoff")
axes[1].set_title("Standardized residual (anomaly score)"); axes[1].legend()
plt.tight_layout(); plt.savefig(C.FIGS / "12_lgbm_actual_vs_expected.png", dpi=150); plt.close()
fr = pd.DataFrame(metrics["flag_rate_by_type_2024"]).T[["baseline", "lgbm"]]
fr.plot.barh(figsize=(8, 4.5), color=["#9aa5b1", "#2a7f62"]); plt.xlabel("share of buildings flagged (2024)"); plt.title("Flag rate by property type: P90 rule vs LightGBM")
plt.tight_layout(); plt.savefig(C.FIGS / "13_flag_rate_baseline_vs_lgbm.png", dpi=150); plt.close()
imp.sort_values().tail(12).plot.barh(figsize=(7, 4.3), color="#5b7db1"); plt.title("LightGBM feature importance (gain)")
plt.tight_layout(); plt.savefig(C.FIGS / "14_lgbm_feature_importance.png", dpi=150); plt.close()
print(json.dumps(metrics, indent=2, default=str)); print("total s", round(time.time() - t0))
