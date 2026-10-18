"""Per-property-type robust scaling of LightGBM residuals.

A single global sigma over-flags property types whose residuals are naturally wider (small, heterogeneous classes).
Here each type gets a robust center (median) and scale (1.4826 * MAD), shrunk toward the global values for small types.
"""
import numpy as np
import pandas as pd


def _robust(r):
    med = np.median(r)
    return med, 1.4826 * np.median(np.abs(r - med))


def fit_type_scaling(prop_type, resid, k: int = 50):
    resid = np.asarray(resid, float)
    g_center, g_sigma = _robust(resid)
    rows = {}
    for t, idx in pd.Series(np.arange(len(resid))).groupby(np.asarray(prop_type)):
        r = resid[idx.values]
        c, s = _robust(r)
        w = len(r) / (len(r) + k)
        rows[t] = {"n": len(r), "center": w * c + (1 - w) * g_center, "sigma": w * s + (1 - w) * g_sigma}
    return pd.DataFrame(rows).T, {"center": g_center, "sigma": g_sigma}


def scale_residuals(prop_type, resid, table, glob):
    pt = pd.Series(np.asarray(prop_type))
    center = pt.map(table["center"]).fillna(glob["center"]).to_numpy()
    sigma = pt.map(table["sigma"]).fillna(glob["sigma"]).to_numpy()
    return (np.asarray(resid, float) - center) / sigma
