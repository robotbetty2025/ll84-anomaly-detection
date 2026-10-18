"""Rule-based benchmark: flag buildings above the 90th percentile of site EUI within their property type."""
import pandas as pd


class PercentileBaseline:
    def __init__(self, q: float = 0.90):
        self.q = q

    def fit(self, df: pd.DataFrame):
        self.thresholds_ = df.groupby("prop_type").site_eui.quantile(self.q)
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Continuous score = EUI / type threshold (>1 means flagged). NaN for types unseen in training."""
        return df.site_eui / df.prop_type.map(self.thresholds_)

    def flag(self, df: pd.DataFrame) -> pd.Series:
        return self.score(df) > 1.0
