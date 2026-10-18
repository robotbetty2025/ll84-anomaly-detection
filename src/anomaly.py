"""Unsupervised anomaly detectors: Isolation Forest (secondary approach) and One-Class SVM (comparison)."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import OneClassSVM

from . import config as C
from .features import CAT_FEATURES, TypeMedianImputer


def iforest_pipeline(num_features, n_estimators=200, seed=C.RANDOM_STATE, contamination=0.10):
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), CAT_FEATURES),
         ("num", "passthrough", list(num_features))]).set_output(transform="pandas")
    return Pipeline([("impute", TypeMedianImputer(num_cols=["log_nat_median_eui"], flag_cols=[])),
                     ("pre", pre),
                     ("model", IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                               random_state=seed, n_jobs=2))])


def ocsvm_pipeline(num_features, nu=0.10):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False), CAT_FEATURES),
         ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), list(num_features))])
    return Pipeline([("impute", TypeMedianImputer(num_cols=["log_nat_median_eui"], flag_cols=[])),
                     ("pre", pre),
                     ("model", OneClassSVM(kernel="rbf", gamma="scale", nu=nu))])


def anomaly_score(pipe, X):
    """Higher = more anomalous."""
    return -pipe.score_samples(X) if hasattr(pipe, "score_samples") else -pipe.decision_function(X)
