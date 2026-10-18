"""LightGBM residual-based anomaly model (primary approach in the proposal)."""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from . import config as C
from .features import CAT_FEATURES, TypeMedianImputer


def make_pipeline(num_features, impute_cols=("log_nat_median_eui",), **lgbm_params):
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), CAT_FEATURES),
         ("num", "passthrough", list(num_features))],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    model = LGBMRegressor(random_state=C.RANDOM_STATE, n_jobs=1, verbose=-1, **lgbm_params)
    return Pipeline([
        ("impute", TypeMedianImputer(num_cols=list(impute_cols), flag_cols=list(impute_cols))),
        ("pre", pre),
        ("model", model),
    ])


def standardized_residual(y_true_log, y_pred_log, sigma):
    return (np.asarray(y_true_log) - np.asarray(y_pred_log)) / sigma
