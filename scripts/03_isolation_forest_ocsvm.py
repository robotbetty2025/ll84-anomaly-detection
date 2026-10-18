"""Step 3: Isolation Forest (tree count chosen by seed stability), One-Class SVM, multi-seed stability."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, numpy as np, pandas as pd
from src import config as C
from src.anomaly import iforest_pipeline, ocsvm_pipeline, anomaly_score
from src.evaluation import build_pool, pairwise_stability
from src.features import CAT_FEATURES, NUM_FEATURES, TARGET, sparse_features

t0 = time.time()
allr = pd.read_parquet(C.PROCESSED / "ll84_all_classified.parquet")
ready = allr[allr.status == "model_ready"]
train = ready[ready.year.isin(C.TRAIN_YEARS)].reset_index(drop=True)
num_features = [c for c in NUM_FEATURES if c not in sparse_features(train, NUM_FEATURES)] + [TARGET]   # features + log EUI
feats = CAT_FEATURES + num_features
pool = build_pool(allr, train.prop_type.unique())
print("train", len(train), "pool", len(pool), "features", feats, flush=True)
out, log = {}, {"features": feats, "pool_size": len(pool)}

# ---- Isolation Forest: smallest tree count whose scores are stable across 5 random seeds
seeds = [11, 22, 33, 44, 55]
stab = {}
for n in (50, 100, 200, 400, 800):
    sc = [anomaly_score(iforest_pipeline(num_features, n, s).fit(train[feats]), pool[feats]) for s in seeds]
    stab[n] = pairwise_stability(sc)
    print("IF trees", n, {k: round(v, 4) for k, v in stab[n].items()}, round(time.time() - t0), "s", flush=True)
    if time.time() - t0 > 100:
        break
log["iforest_stability_by_trees"] = stab
chosen = next((n for n, m in stab.items() if m["mean_spearman"] >= 0.99 and m["mean_top10_jaccard"] >= 0.85), max(stab))
log["iforest_trees_chosen"] = chosen
print("chosen trees", chosen, flush=True)
if_pipe = iforest_pipeline(num_features, chosen, C.RANDOM_STATE).fit(train[feats])
out["if_score"] = anomaly_score(if_pipe, pool[feats])
log["iforest_stability_final"] = stab[chosen]
joblib.dump({"pipeline": if_pipe, "features": feats}, C.MODELS / "isolation_forest.joblib")

# ---- One-Class SVM on a scaled random subset (comparison model)
sub = train.sample(8000, random_state=C.RANDOM_STATE)
oc = ocsvm_pipeline(num_features).fit(sub[feats])
out["ocsvm_score"] = anomaly_score(oc, pool[feats])
joblib.dump({"pipeline": oc, "features": feats}, C.MODELS / "ocsvm.joblib")
print("ocsvm done", round(time.time() - t0), "s", flush=True)

res = pool[["property_id", "year", "status"]].copy()
for k, v in out.items():
    res[k] = v
res.to_parquet(C.PROCESSED / "pool_scores_unsupervised.parquet", index=False)
(C.REPORTS / "week5_unsupervised_log.json").write_text(json.dumps(log, indent=2, default=str))
print(json.dumps(log, indent=2, default=str)); print("total", round(time.time() - t0), "s")
