"""Feature engineering and the leakage-safe sklearn preprocessing used by the models."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C

FUEL_GROUPS = {
    "share_elec": ["e_grid_kbtu", "e_onsite_kbtu"],
    "share_gas": ["gas_kbtu"],
    "share_steam": ["steam_kbtu", "dhw_kbtu", "dchw_kbtu"],
    "share_oil": ["oil1_kbtu", "oil2_kbtu", "oil4_kbtu", "oil56_kbtu", "diesel_kbtu", "kerosene_kbtu"],
    "share_other": ["propane_kbtu"],
}
CAT_FEATURES = ["prop_type", "borough"]
NUM_FEATURES = ["log_gfa", "building_age", "n_buildings", "occupancy", "log_nat_median_eui",
                "share_elec", "share_gas", "share_steam", "share_oil", "share_other",
                "weekly_hours", "default_values", "estimated_energy"]
TARGET = "log_site_eui"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_gfa"] = np.log(d.gfa)
    d[TARGET] = np.log(d.site_eui.where(d.site_eui > 0))
    d["log_nat_median_eui"] = np.log(d.nat_median_site_eui.where(d.nat_median_site_eui > 0))
    # an unreported fuel means zero use of that fuel; shares are only undefined when no fuel is reported at all
    parts = {k: d[cols].clip(lower=0).fillna(0).sum(axis=1) for k, cols in FUEL_GROUPS.items()}
    total = pd.concat(parts, axis=1).sum(axis=1).replace(0, np.nan)
    for k, s in parts.items():
        d[k] = (s / total).clip(0, 1)
    hrs = d[["hrs_office", "hrs_medoffice", "hrs_finoffice", "hrs_restaurant"]]
    d["weekly_hours"] = hrs.where((hrs > 0) & (hrs <= 168)).bfill(axis=1).iloc[:, 0]
    d["eui_to_median_ratio"] = d.site_eui / d.nat_median_site_eui.where(d.nat_median_site_eui > 0)
    return d


def sparse_features(df: pd.DataFrame, features, max_missing=C.MAX_MISSING_FRAC) -> dict:
    """Return {feature: missing_fraction} for features exceeding the missingness limit (proposal rule: drop >40%)."""
    frac = df[features].isna().mean()
    return frac[frac > max_missing].round(3).to_dict()


class TypeMedianImputer(BaseEstimator, TransformerMixin):
    """Impute numeric columns with the property-type median learned on the training fold only.

    Adds a binary `<col>_was_imputed` flag for each column in `flag_cols` (proposal requirement).
    """

    def __init__(self, num_cols=(), flag_cols=(), type_col="prop_type"):
        self.num_cols = num_cols
        self.flag_cols = flag_cols
        self.type_col = type_col

    def fit(self, X, y=None):
        cols = list(self.num_cols)
        self.global_ = X[cols].median()
        self.by_type_ = X.groupby(self.type_col)[cols].median()
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.num_cols:
            miss = X[c].isna()
            fill = X[self.type_col].map(self.by_type_[c]).fillna(self.global_[c])
            X[c] = X[c].fillna(fill)
            if c in self.flag_cols:
                X[f"{c}_was_imputed"] = miss.astype(int)
        return X
