"""Loading, de-duplication, scope filtering and plausibility screening for LL84 data."""
import numpy as np
import pandas as pd

from . import config as C

NA_STRINGS = ["Not Available", "Not Applicable", "Not Applicable: Standalone Property", "Unable to Check (not enough data)"]


def load_raw(path=None) -> pd.DataFrame:
    path = path or C.find_raw_csv()
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in C.RAW_COLS if c not in header]
    if missing:
        raise KeyError(f"Expected columns not in file: {missing}")
    df = pd.read_csv(path, usecols=list(C.RAW_COLS), low_memory=False, na_values=NA_STRINGS[:3])
    df = df.rename(columns=C.RAW_COLS)
    for c in C.NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["report_submitted"] = pd.to_datetime(df["report_submitted"], errors="coerce")
    for c in ("default_values", "estimated_energy"):
        df[c] = df[c].astype(str).str.strip().str.lower().map(
            {"yes": 1, "true": 1, "no": 0, "false": 0}).astype("float")
    return df


def resolve_duplicates(df: pd.DataFrame):
    """One row per Property ID per year: keep the most complete, then most recently submitted."""
    key_cols = ["site_eui", "site_energy_kbtu", "nat_median_site_eui", "year_built", "occupancy", "gfa_largest_use"]
    d = df.copy()
    d["_completeness"] = d[key_cols].notna().sum(axis=1)
    d = d.sort_values(["property_id", "year", "_completeness", "report_submitted"],
                      ascending=[True, True, False, False])
    out = d.drop_duplicates(["property_id", "year"], keep="first").drop(columns="_completeness")
    return out.sort_index(), len(df) - len(out)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Property GFA implied by EPA's own metric: site energy / site EUI (the 'calculated GFA' columns are ~99% empty).
    ok = (d.site_eui > 0) & (d.site_energy_kbtu > 0)
    d["gfa"] = np.where(ok, d.site_energy_kbtu / d.site_eui, np.nan)
    d["building_age"] = d.year - d.year_built
    # year-over-year EUI ratio for the same property (used for EDA and the proxy-label benchmark later)
    d = d.sort_values(["property_id", "year"])
    prev = d.groupby("property_id")["site_eui"].shift(1)
    prev_year = d.groupby("property_id")["year"].shift(1)
    d["eui_yoy_ratio"] = np.where((prev > 0) & (d.year - prev_year == 1), d.site_eui / prev, np.nan)
    return d.sort_index()


def classify_rows(df: pd.DataFrame):
    """Assign each row to out_of_scope / dq_anomaly / model_ready with a reason.

    Per the proposal, implausible records are isolated as a 'data-quality anomaly' class, not discarded.
    """
    d = df.copy()
    n_bldg = d.groupby("prop_type")["property_id"].nunique()
    rare = set(n_bldg[n_bldg < C.MIN_BUILDINGS_PER_TYPE].index)
    status = pd.Series("model_ready", index=d.index)
    reason = pd.Series("", index=d.index)

    def mark(mask, st, why):
        m = mask & (status == "model_ready")
        status.loc[m] = st
        reason.loc[m] = why

    mark(d.construction_status.fillna("Existing") != "Existing", "out_of_scope", "not_existing_building")
    mark(d.prop_type.isna(), "out_of_scope", "missing_property_type")
    mark(d.prop_type.isin(C.EXCLUDED_TYPES), "out_of_scope", "excluded_type")
    mark(d.prop_type.isin(rare), "out_of_scope", f"type_under_{C.MIN_BUILDINGS_PER_TYPE}_buildings")
    mark(d.site_eui.isna() | d.site_energy_kbtu.isna(), "out_of_scope", "missing_required_energy")
    # --- data-quality anomaly class (kept, not discarded) ---
    mark((d.site_eui < C.EUI_MIN) | (d.site_eui > C.EUI_MAX), "dq_anomaly", "site_eui_out_of_range")
    mark(d.gfa.isna() | (d.gfa < C.GFA_MIN), "dq_anomaly", "gfa_below_25k_or_invalid")
    mark((d.year_built < C.YEAR_BUILT_MIN) | (d.year_built > d.year) | d.year_built.isna(),
         "dq_anomaly", "year_built_invalid")
    d["status"], d["dq_reason"] = status, reason
    return d
