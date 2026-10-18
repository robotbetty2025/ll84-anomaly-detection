"""Step 1: load raw LL84, de-duplicate, derive fields, classify rows, engineer features, save."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config as C
from src.cleaning import load_raw, resolve_duplicates, add_derived, classify_rows
from src.features import build_features, CAT_FEATURES, NUM_FEATURES, sparse_features

C.PROCESSED.mkdir(parents=True, exist_ok=True)
raw = load_raw()
log = {"raw_rows": len(raw), "raw_by_year": raw.year.value_counts().sort_index().to_dict()}
d, n_dup = resolve_duplicates(raw)
log["duplicate_property_years_removed"] = n_dup
d = add_derived(d)
d = classify_rows(d)
d = build_features(d)
log["status_counts"] = d.status.value_counts().to_dict()
log["status_by_year"] = d.groupby(["year", "status"]).size().unstack(fill_value=0).to_dict("index")
log["scope_reasons"] = d[d.status != "model_ready"].groupby(["status", "dq_reason"]).size().to_dict().__repr__()
ready = d[d.status == "model_ready"]
log["model_ready_by_year"] = ready.year.value_counts().sort_index().to_dict()
log["model_ready_buildings"] = int(ready.property_id.nunique())
log["property_types_in_model"] = int(ready.prop_type.nunique())
log["feature_missing_gt40pct_in_model_ready"] = sparse_features(ready, NUM_FEATURES + CAT_FEATURES)
log["feature_missing_frac_all"] = ready[NUM_FEATURES + CAT_FEATURES].isna().mean().round(3).to_dict()
d.to_parquet(C.PROCESSED / "ll84_all_classified.parquet", index=False)
ready.to_parquet(C.PROCESSED / "ll84_model_ready.parquet", index=False)
d[d.status == "dq_anomaly"].to_parquet(C.PROCESSED / "ll84_dq_anomalies.parquet", index=False)
C.REPORTS.mkdir(exist_ok=True)
(C.REPORTS / "dataset_build_log.json").write_text(json.dumps(log, indent=2, default=str))
print(json.dumps(log, indent=2, default=str))
