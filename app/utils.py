"""Shared loading, scoring and plotting helpers for the Streamlit app."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # so pickled models can import src.*

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.scoring import scale_residuals

DATA = Path(__file__).resolve().parent / "data"
FIGS = Path(__file__).resolve().parent / "figures"
EVAL_YEAR = 2024
KBTU_PER_KWH, KBTU_PER_THERM, KBTU_PER_GAL_OIL2 = 3.412, 100.0, 138.5

FEATURE_LABELS = {
    "prop_type": "Property type", "borough": "Borough", "log_gfa": "Floor area", "building_age": "Building age",
    "n_buildings": "Number of buildings", "occupancy": "Occupancy", "log_nat_median_eui": "National median EUI (peer benchmark)",
    "share_elec": "Electricity share", "share_gas": "Natural gas share", "share_steam": "District steam share",
    "share_oil": "Fuel oil share", "share_other": "Other fuel share", "default_values": "Default values flag",
    "estimated_energy": "Estimated energy flag", "log_nat_median_eui_was_imputed": "Benchmark was imputed",
}


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    return joblib.load(DATA / "model_bundle.joblib")


@st.cache_data(show_spinner="Loading scored buildings...")
def load_scored() -> pd.DataFrame:
    d = pd.read_parquet(DATA / "scored_2024.parquet")
    d["lgbm_score"] = d.score_pertype.abs()
    d["direction"] = np.where(d.score_pertype >= 0, "above expected", "below expected")
    d["ratio_actual_expected"] = d.site_eui / d.pred_eui
    fuels = d[["share_elec", "share_gas", "share_steam", "share_oil"]].fillna(0)
    d["dominant_fuel"] = fuels.idxmax(axis=1).str.replace("share_", "", regex=False).map(
        {"elec": "Electricity", "gas": "Natural gas", "steam": "District steam", "oil": "Fuel oil"})
    return d


@st.cache_data
def load_results() -> dict:
    return json.loads((DATA / "results.json").read_text())


def shap_contributions(feature_row: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP values (LightGBM pred_contrib) in ln(EUI) units for one building."""
    b = load_bundle()
    pipe = b["lgbm"]
    Xt = pipe[:-1].transform(feature_row[b["features"]])
    contrib = pipe.named_steps["model"].booster_.predict(Xt, pred_contrib=True)[0]
    out = pd.DataFrame({"feature": Xt.columns, "contribution": contrib[:-1]})
    out["label"] = out.feature.map(FEATURE_LABELS).fillna(out.feature)
    out["value"] = [Xt.iloc[0][f] for f in out.feature]
    return out.assign(base=float(contrib[-1]))


def waterfall_figure(contrib: pd.DataFrame, title: str, top_n: int = 8) -> go.Figure:
    c = contrib.reindex(contrib.contribution.abs().sort_values(ascending=False).index)
    shown, rest = c.head(top_n), c.iloc[top_n:]
    labels = ["Typical building (model average)"] + list(shown.label) + (["All other features"] if len(rest) else []) + ["Expected EUI"]
    vals = [contrib.base.iloc[0]] + list(shown.contribution) + ([rest.contribution.sum()] if len(rest) else [])
    total = float(np.sum(vals))
    fig = go.Figure(go.Waterfall(orientation="h", y=labels, x=vals + [0], measure=["absolute"] + ["relative"] * (len(vals) - 1) + ["total"],
                                 text=[f"{np.exp(vals[0]):.0f} kBtu/ft²"] + [f"{v:+.2f}" for v in vals[1:]] + [f"{np.exp(total):.0f} kBtu/ft²"],
                                 increasing={"marker": {"color": "#d9822b"}}, decreasing={"marker": {"color": "#4c78a8"}},
                                 totals={"marker": {"color": "#2a7f62"}}, textposition="outside"))
    fig.update_layout(title=title, height=430, margin=dict(l=10, r=30, t=50, b=40), yaxis=dict(autorange="reversed"),
                      xaxis_title="ln(EUI): orange pushes expected EUI up, blue pushes it down")
    return fig


def build_features(prop_type, borough, gfa, year_built, occupancy, n_buildings, kwh, therms, steam_kbtu, oil_gal):
    """Turn user inputs into one model-ready feature row plus the building's site EUI."""
    b = load_bundle()
    parts = {"share_elec": kwh * KBTU_PER_KWH, "share_gas": therms * KBTU_PER_THERM,
             "share_steam": steam_kbtu, "share_oil": oil_gal * KBTU_PER_GAL_OIL2, "share_other": 0.0}
    total = sum(parts.values())
    eui = total / gfa if gfa > 0 else np.nan
    nat = b["nat_median_by_type"].get(prop_type, np.nan)
    row = {"prop_type": prop_type, "borough": borough, "log_gfa": np.log(gfa), "building_age": EVAL_YEAR - year_built,
           "n_buildings": n_buildings, "occupancy": occupancy, "log_nat_median_eui": np.log(nat) if nat and nat > 0 else np.nan,
           "default_values": 0.0, "estimated_energy": 0.0,
           "log_site_eui": np.log(eui) if eui and eui > 0 else np.nan}
    row.update({k: (v / total if total > 0 else np.nan) for k, v in parts.items()})
    return pd.DataFrame([row]), eui


def score_building(frame: pd.DataFrame, eui: float) -> dict:
    b = load_bundle()
    pred_log = float(b["lgbm"].predict(frame[b["features"]])[0])
    resid = float(np.log(eui) - pred_log)
    z = float(scale_residuals(frame.prop_type, [resid], b["type_scaling"], b["global_scaling"])[0])
    p90 = b["baseline_thresholds"].get(frame.prop_type.iloc[0], np.nan)
    if_raw = float(-b["iforest"].score_samples(frame[b["iforest_features"]])[0])
    ref = b["if_ref"]
    return {"expected_eui": float(np.exp(pred_log)), "eui": float(eui), "z": z, "flag_lgbm": z >= b["thr_lgbm"],
            "p90": float(p90), "flag_rule": bool(eui > p90), "if_score": if_raw, "if_pct": float((ref < if_raw).mean() * 100),
            "flag_if": if_raw >= b["thr_if"]}
