import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import utils as U

st.set_page_config(page_title="Building Energy Anomaly Detector", page_icon="🏢", layout="wide")
st.title("Building Energy Anomaly Detector")
st.caption("NYC Local Law 84 benchmarking data, calendar years 2022 to 2024. Model trained on 2022-2023, evaluated on 2024.")

bundle, scored, R = U.load_bundle(), U.load_scored(), U.load_results()
ALGOS = {"LightGBM residual (per-type scaled)": ("lgbm_score", "lgbm_flag"),
         "Isolation Forest": ("if_score", "if_flag"),
         "Rule-based: above 90th percentile of type": ("baseline_ratio", "baseline_flag")}

# ------------------------------------------------------------------ sidebar filters (used by the Explore tab)
with st.sidebar:
    st.header("Explore filters")
    st.caption("These controls apply to the 'Explore Portfolio Rankings' tab.")
    algo = st.radio("Anomaly score", list(ALGOS), help="Which method ranks the buildings.")
    types = st.multiselect("Property type", sorted(scored.prop_type.unique()), default=["Multifamily Housing", "Office", "Hotel", "K-12 School"])
    boroughs = st.multiselect("Borough", sorted(scored.borough.dropna().unique()), default=sorted(scored.borough.dropna().unique()))
    gmax = int(scored.gfa.quantile(0.995))
    gfa_rng = st.slider("Floor area (ft²)", 25_000, gmax, (25_000, gmax), step=5_000)
    yb_rng = st.slider("Year built", 1850, 2024, (1850, 2024))
    fuels = st.multiselect("Dominant fuel", ["Electricity", "Natural gas", "District steam", "Fuel oil"], default=["Electricity", "Natural gas", "District steam", "Fuel oil"])
    only_flagged = st.checkbox("Only show flagged buildings (top 10% by this method)", value=False)
    score_col, flag_col = ALGOS[algo]
    smin = st.slider("Minimum anomaly score", float(scored[score_col].min()), float(scored[score_col].quantile(0.999)), float(scored[score_col].min()))

tab1, tab2, tab3, tab4 = st.tabs(["Problem & Approach", "Results & Performance", "Explore Portfolio Rankings", "Score Your Own Building"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.header("The problem")
    st.markdown(
        "Cities such as New York require large buildings to report their annual energy use. Most programs then flag "
        "buildings with a simple rule, for example 'above the 90th percentile for its property type'. That rule has two weaknesses:\n\n"
        "- It treats very different buildings alike. A 24/7 data center or hospital is *supposed* to use a lot of energy per square foot, "
        "while a small office using the same amount may be wasting a lot.\n"
        "- It looks at one number at a time, so it misses how size, age, fuel type and use combine to set what a building should use.\n\n"
        "**This project asks a different question:** given a building's type, size, age, location and fuel mix, how much energy would we "
        "*expect* it to use, and how far is its actual use from that expectation?")
    st.header("Approach")
    st.markdown(
        "1. **Data.** About 103,000 building-year records from NYC Local Law 84 (2022 to 2024), plus the EPA ENERGY STAR national median EUI "
        "for each property type as a peer benchmark. EUI (energy use intensity) is annual energy per square foot.\n"
        "2. **Cleaning.** Duplicates are removed. Implausible records (for example an impossible EUI or floor area) are kept as their own "
        "data-quality group instead of being deleted, since a reporting error is itself something an analyst wants to see.\n"
        "3. **Expected-EUI model.** A gradient boosted model (LightGBM) learns what buildings like this one normally use. "
        "The anomaly score is how many 'typical misses' the actual EUI is from the expected EUI, scaled separately for each property type.\n"
        "4. **Comparison methods.** A rule-based percentile cutoff, an Isolation Forest, and a One-Class SVM.\n"
        "5. **Explanations.** SHAP values show which factors pushed a building's expected EUI up or down.")
    st.header("What the score means")
    st.info("A high score means a building uses much more (or less) energy than similar buildings *with the same characteristics*. "
            "It is a prompt to investigate, not a verdict. The cause could be inefficiency, an unusual use, or a reporting error.")
    st.header("Limits to keep in mind")
    st.markdown(
        "- The data is self-reported, and some records use default or estimated values.\n"
        "- Annual data cannot show seasonal or short-term problems.\n"
        "- Buildings under about 25,000 ft² are not covered.\n"
        "- Weekly operating hours were too sparse (about 85% missing) to use as a model input.\n"
        "- There are no true anomaly labels, so performance is judged with proxy labels and a manual audit.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.header("How well does the expected-EUI model predict?")
    reg = R["regression_eval_2024"]
    t = pd.DataFrame({"Property-type median": reg["property_type_median"], "Linear regression": reg["linear_regression"], "LightGBM": reg["lightgbm"]}).T
    t = t.rename(columns={"rmse_log": "RMSE (ln EUI)", "r2_log": "R² (ln EUI)", "mape_eui": "MAPE (EUI)", "rmse_eui": "RMSE (kBtu/ft²)"})
    st.dataframe(t[["RMSE (ln EUI)", "R² (ln EUI)", "MAPE (EUI)", "RMSE (kBtu/ft²)"]].style.format({"RMSE (ln EUI)": "{:.3f}", "R² (ln EUI)": "{:.2f}", "MAPE (EUI)": "{:.0%}", "RMSE (kBtu/ft²)": "{:.1f}"}))
    st.caption("Evaluated on the 2024 hold-out year, which the model never saw during training or tuning.")

    st.header("How well does each method find unusual buildings?")
    st.write("There are no ground-truth anomaly labels, so methods are compared against **proxy labels**: records that fail plausibility screens, "
             "year-over-year EUI jumps of more than 3x, and extreme ENERGY STAR scores (1 to 5 or 95 to 100).")
    def bench(key):
        b = pd.DataFrame(R["benchmark"][key]).set_index("method")
        return b[["pr_auc", "precision@5%", "precision@10%", "recall@10%"]].rename(columns={"pr_auc": "PR-AUC"})
    c1, c2 = st.columns(2)
    c1.markdown(f"**All proxy labels** (base rate {R['proxy_label_counts']['any_proxy'] / R['proxy_label_counts']['pool']:.0%})")
    c1.dataframe(bench("any_proxy").style.format("{:.2f}"))
    c2.markdown(f"**Excluding plausibility failures** (base rate {R['proxy_label_counts']['any_excl_plausibility'] / R['proxy_label_counts']['pool']:.0%})")
    c2.dataframe(bench("any_excl_plausibility").style.format("{:.2f}"))
    st.warning("Read this carefully: the proxy labels are built from EUI itself (the ENERGY STAR score is computed from EUI, and so are the plausibility "
               "screens), so they favor any method that ranks raw EUI extremes. A simple two-sided percentile rule therefore does about as well as "
               "the LightGBM score here. The residual model's advantage is meant to be the *explanation* (what is unusual given the building's "
               "characteristics), which the proxy labels cannot measure. The manual audit is the fairer test.")
    st.image(str(U.FIGS / "16_proxy_benchmark.png"))

    st.header("Is the ranking stable?")
    s1, s2 = R["stability_lightgbm"], R["stability_iforest"]
    st.markdown(f"Across 5 random seeds, LightGBM rankings correlate at a mean Spearman of **{s1['mean_spearman']:.3f}** (top-10% overlap {s1['mean_top10_jaccard']:.2f}); "
                f"Isolation Forest at **{s2['mean_spearman']:.3f}** (top-10% overlap {s2['mean_top10_jaccard']:.2f}).")
    st.image(str(U.FIGS / "17_model_agreement.png"))

    st.header("What drives the expected EUI? (SHAP)")
    cc1, cc2 = st.columns(2)
    cc1.image(str(U.FIGS / "19_shap_global_importance.png"))
    cc2.image(str(U.FIGS / "18_shap_summary.png"))
    st.header("Per-type scaling")
    st.write("A single global scale over-flagged property types whose residuals are naturally wider. Scaling per type evens out the flag rates.")
    st.image(str(U.FIGS / "15_flag_rate_per_type_scaling.png"))
    st.header("Actual vs expected EUI (2024 hold-out)")
    st.image(str(U.FIGS / "12_lgbm_actual_vs_expected.png"))

# ------------------------------------------------------------------ tab 3
with tab3:
    st.header("Explore portfolio rankings (2024 hold-out buildings)")
    d = scored.copy()
    d["score"] = d[score_col]
    d["rank"] = d.score.rank(ascending=False, method="first").astype(int)
    m = (d.prop_type.isin(types) & d.borough.isin(boroughs) & d.gfa.between(*gfa_rng) & d.year_built.between(*yb_rng)
         & d.dominant_fuel.isin(fuels) & (d.score >= smin))
    if only_flagged:
        m &= d[flag_col]
    f = d[m].sort_values("score", ascending=False)
    k1, k2, k3 = st.columns(3)
    k1.metric("Buildings matching filters", f"{len(f):,}")
    k2.metric("Flagged (top 10% overall)", f"{int(f[flag_col].sum()):,}")
    k3.metric("Median EUI of selection", f"{f.site_eui.median():.0f} kBtu/ft²" if len(f) else "n/a")
    if f.empty:
        st.info("No buildings match these filters. Loosen a filter in the sidebar.")
    else:
        show = f[["rank", "property_name", "address", "borough", "prop_type", "year_built", "gfa", "site_eui", "pred_eui", "ratio_actual_expected", "direction", "score", flag_col]].rename(columns={
            "rank": "Rank", "property_name": "Name", "address": "Address", "borough": "Borough", "prop_type": "Type", "year_built": "Year built",
            "gfa": "Floor area (ft²)", "site_eui": "Actual EUI", "pred_eui": "Expected EUI", "ratio_actual_expected": "Actual / expected",
            "direction": "Direction", "score": "Score", flag_col: "Flagged"})
        st.dataframe(show.head(500).style.format({"Floor area (ft²)": "{:,.0f}", "Actual EUI": "{:.1f}", "Expected EUI": "{:.1f}", "Actual / expected": "{:.2f}", "Score": "{:.2f}"}), hide_index=True, height=320)
        st.caption("Showing the 500 highest-scoring buildings in the selection. The CSV download contains every matching building.")
        st.download_button("Download filtered results (CSV)", show.to_csv(index=False).encode(), file_name="filtered_building_rankings.csv", mime="text/csv")

        st.subheader("Actual vs expected EUI")
        pts = f.sample(min(len(f), 6000), random_state=1) if len(f) > 6000 else f
        sel_label = st.selectbox("Inspect a building (top 300 in the selection)", [f"#{r.rank}  {r.property_name} ({r.address})" for r in f.head(300).itertuples()])
        sel = f.head(300).iloc[[f"#{r.rank}  {r.property_name} ({r.address})" for r in f.head(300).itertuples()].index(sel_label)]
        fig = px.scatter(pts, x="pred_eui", y="site_eui", color=flag_col, log_x=True, log_y=True, opacity=0.5, hover_name="property_name",
                         hover_data={"prop_type": True, "borough": True, flag_col: False}, color_discrete_map={True: "#d9822b", False: "#9aa5b1"},
                         labels={"pred_eui": "Expected EUI (kBtu/ft²)", "site_eui": "Actual EUI (kBtu/ft²)", flag_col: "Flagged"})
        lim = [max(1, min(pts.pred_eui.min(), pts.site_eui.min())), max(pts.pred_eui.max(), pts.site_eui.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="black", dash="dash", width=1), name="actual = expected", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[sel.pred_eui], y=[sel.site_eui], mode="markers", marker=dict(size=14, color="#c0392b", symbol="star"), name="selected"))
        fig.update_layout(height=450, margin=dict(t=10))
        st.plotly_chart(fig)

        st.subheader(f"Drill-down: {sel.property_name}")
        st.write(f"{sel.address}, {sel.borough} | {sel.prop_type} | built {int(sel.year_built)} | {sel.gfa:,.0f} ft² | "
                 f"actual EUI **{sel.site_eui:.0f}**, expected **{sel.pred_eui:.0f}** kBtu/ft² ({sel.ratio_actual_expected:.2f}x, {sel.direction})")
        pc1, pc2 = st.columns(2)
        ps = bundle["peer_stats"].loc[sel.prop_type]
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=["10th", "25th", "median", "75th", "90th"], y=[ps["10%"], ps["25%"], ps["50%"], ps["75%"], ps["90%"]], marker_color="#9aa5b1", name=f"{sel.prop_type} peers (2022-23)"))
        fig_p.add_hline(y=sel.site_eui, line_color="#c0392b", annotation_text=f"this building: {sel.site_eui:.0f}")
        fig_p.update_layout(title=f"EUI vs all {int(ps['count']):,} {sel.prop_type} records", yaxis_title="Site EUI (kBtu/ft²)", xaxis_title="percentile of peers", height=430, margin=dict(t=50))
        pc1.plotly_chart(fig_p)
        frame = pd.DataFrame([sel])[bundle["features"]]
        pc2.plotly_chart(U.waterfall_figure(U.shap_contributions(frame), "Why this expected EUI? (SHAP)"))
        st.caption("The waterfall explains the model's *expected* EUI. The anomaly is the gap between that expectation and the actual EUI.")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.header("Score your own building")
    st.write("Enter a real or hypothetical building. The app computes its EUI, the model's expected EUI, and an anomaly score. "
             "Weekly operating hours are not used because that field is too sparse in the data.")
    with st.form("own_building"):
        a, b = st.columns(2)
        ptype = a.selectbox("Property type", bundle["prop_types"], index=bundle["prop_types"].index("Office"))
        boro = b.selectbox("Borough", bundle["boroughs"], index=0)
        gfa = a.number_input("Gross floor area (ft²)", 25_000, 5_000_000, 150_000, step=5_000)
        yb = b.number_input("Year built", 1850, 2024, 1985)
        occ = a.slider("Occupancy (%)", 0, 100, 95)
        nb = b.number_input("Number of buildings", 1, 50, 1)
        st.markdown("**Annual energy use**")
        e1, e2, e3, e4 = st.columns(4)
        kwh = e1.number_input("Electricity (kWh)", 0.0, 1e9, 2_500_000.0, step=50_000.0)
        therms = e2.number_input("Natural gas (therms)", 0.0, 1e8, 40_000.0, step=1_000.0)
        steam = e3.number_input("District steam (kBtu)", 0.0, 1e10, 0.0, step=100_000.0)
        oil = e4.number_input("Fuel oil #2 (gallons)", 0.0, 1e7, 0.0, step=1_000.0)
        go_btn = st.form_submit_button("Score this building")
    if go_btn:
        frame, eui = U.build_features(ptype, boro, gfa, yb, occ, nb, kwh, therms, steam, oil)
        if not np.isfinite(eui) or eui <= 0:
            st.error("Enter some energy use to compute an EUI.")
        else:
            r = U.score_building(frame, eui)
            if eui < 5 or eui > 1000:
                st.warning("This EUI is outside the 5 to 1,000 kBtu/ft² plausibility range used in the project. Records like this are treated as possible data errors.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Actual site EUI", f"{r['eui']:.0f}")
            m2.metric("Expected EUI (LightGBM)", f"{r['expected_eui']:.0f}", f"{(r['eui'] / r['expected_eui'] - 1):+.0%} vs expected", delta_color="off")
            m3.metric("Anomaly score (per-type)", f"{r['z']:+.2f}")
            m4.metric(f"90th pct of {ptype}", f"{r['p90']:.0f}")
            verdicts = [("LightGBM residual", r["flag_lgbm"]), ("Isolation Forest", r["flag_if"]), ("90th-percentile rule", r["flag_rule"])]
            st.write("**Flags:** " + " | ".join(f"{n}: {'FLAGGED' if v else 'not flagged'}" for n, v in verdicts) +
                     f"  (Isolation Forest percentile among 2024 buildings: {r['if_pct']:.0f})")
            st.plotly_chart(U.waterfall_figure(U.shap_contributions(frame), "What shaped the expected EUI (SHAP)"))
            st.caption("A positive score means higher use than expected for similar buildings; a negative score means lower. Flags use the top 10% of 2024 buildings as the cutoff.")
