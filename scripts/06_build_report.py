"""Step 6: build the written Project Overview PDF from the saved results (no numbers are typed by hand).

Re-run after finishing the audit (scripts/05_audit_results.py) and after editing docs/report_config.json.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
from src import config as C

R, DOCS = C.REPORTS, C.ROOT / "docs"
cfg = json.loads((DOCS / "report_config.json").read_text())
b = json.loads((R / "dataset_build_log.json").read_text())
m = json.loads((R / "final_metrics.json").read_text())
w = json.loads((R / "week4_metrics.json").read_text())
u = json.loads((R / "week5_unsupervised_log.json").read_text())
audit = json.loads((R / "audit_results.json").read_text()) if (R / "audit_results.json").exists() else None
allr = pd.read_parquet(C.PROCESSED / "ll84_all_classified.parquet")
ready = allr[allr.status == "model_ready"]

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10, leading=14, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=colors.HexColor("#1f5c49"), spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=colors.HexColor("#1f5c49"), spaceBefore=10, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=BODY, fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=10)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#1f5c49"))
CEN = ParagraphStyle("cen", parent=BODY, alignment=TA_CENTER, fontSize=11, leading=16)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.3, leading=10.2, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=2, spaceAfter=3)
PEND = ParagraphStyle("pend", parent=BODY, backColor=colors.HexColor("#fff3cd"), borderColor=colors.HexColor("#d9822b"), borderWidth=0.8, borderPadding=6, spaceBefore=4, spaceAfter=10)

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))
def bullets(items):
    for t in items: story.append(Paragraph(t, BUL, bulletText="•"))
def fig(name, cap, width=6.3, max_h=4.3):
    path = C.FIGS / name
    iw, ih = Image.open(path).size
    wd = width * inch; ht = wd * ih / iw
    if ht > max_h * inch: ht = max_h * inch; wd = ht * iw / ih
    story.append(KeepTogether([RLImage(str(path), width=wd, height=ht), Paragraph(cap, CAP)]))
def table(rows, widths, header=True, align_right_from=1, font=8.3):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[x * inch for x in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4c0")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcece6")))
    t.setStyle(TableStyle(st)); story.append(t); story.append(Spacer(1, 4))
f0 = lambda x: f"{x:,.0f}"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; pc = lambda x: f"{x:.0%}"; pc1 = lambda x: f"{x:.1%}"
NAMES = {"P90 rule (EUI / type P90)": "Percentile rule: above P90 (proposal baseline)", "P10/P90 rule (two-sided)": "Percentile rule: outside P10 to P90 (two-sided)",
         "EUI / national median": "Ratio to national median EUI", "LightGBM |resid| (global sigma)": "LightGBM residual, one global scale",
         "LightGBM |resid| (per-type sigma)": "LightGBM residual, per-type scale (final)", "Isolation Forest": "Isolation Forest", "One-Class SVM": "One-Class SVM"}

# ---------------------------------------------------------------- title page
story += [Spacer(1, 1.6 * inch), Paragraph("Building Energy Benchmarking Anomaly Detection", TITLE), Spacer(1, 6),
          Paragraph("Ranking NYC commercial and multifamily buildings by unusual energy use with machine learning", CEN), Spacer(1, 40),
          Paragraph(f"<b>Machine Learning Project Overview</b><br/>{cfg['course']}", CEN), Spacer(1, 18),
          Paragraph(f"{cfg['student']}<br/>Mentor: {cfg['mentor']}<br/>{cfg['submission_date']}", CEN), Spacer(1, 40),
          Paragraph(f"Live application: {cfg['app_url']}<br/>Code repository: {cfg['repo_url']}<br/>Video walkthrough: {cfg['video_url']}", CEN), PageBreak()]

# ---------------------------------------------------------------- abstract
t = w["lgbm_test_2024"]; reg = m["regression_eval_2024"]
P("Abstract", H1)
P(f"Energy benchmarking programs such as New York City's Local Law 84 usually flag buildings with static percentile cutoffs. Those rules treat very different "
  f"buildings alike and ignore how size, age, fuel mix and use combine to set a building's expected energy use. This project builds a machine learning system "
  f"that estimates how much energy a building <i>should</i> use given its characteristics and ranks buildings by how far actual use departs from that expectation. "
  f"Using {f0(b['raw_rows'])} building-year records from LL84 (calendar years 2022 to 2024) and the EPA ENERGY STAR national median EUI as a peer reference, I trained a "
  f"LightGBM regression on 2022 to 2023 data and scored 2024 buildings with a per-property-type standardized residual. The expected-EUI model explains "
  f"{pc(t['r2_log'])} of the variance in log site EUI on the 2024 hold-out (property-type medians: {pc(reg['property_type_median']['r2_log'])}; linear regression: {pc(reg['linear_regression']['r2_log'])}). "
  f"The method is compared with percentile rules, an Isolation Forest and a One-Class SVM using proxy labels, seed stability and SHAP explanations, and is delivered as an interactive four-page Streamlit application. "
  f"A main finding is that proxy labels built from EUI favor simple EUI-extreme rules, so the conditional (explainable) value of the residual model has to be judged by a manual audit rather than proxy metrics alone.")

# ---------------------------------------------------------------- 1 problem
P("1. Problem, Objective and Scope", H1)
P("Municipal disclosure programs and benchmarking tools mostly rely on static percentile cutoffs or simple single-variable regressions. This produces two recurring failure modes. "
  "<b>False positives and negatives:</b> legitimately energy-intensive buildings (for example 24/7 data centers or hospitals) are penalized, while inefficient smaller buildings can look acceptable per square foot. "
  "<b>Uncaptured interactions:</b> cutoffs evaluate one driver at a time and cannot model how floor area, vintage, fuel mix and property type jointly determine a baseline.")
P("<b>Objective.</b> Build an unsupervised and semi-supervised machine learning system that flags commercial and multifamily buildings whose Energy Use Intensity (EUI, annual site energy per square foot) "
  "departs significantly from peer expectations after controlling for legitimate drivers. The deliverable is a continuous, ranked anomaly score per building, surfaced through an interactive interface so that benchmarking analysts, energy managers "
  "and compliance reviewers can prioritize buildings for investigation, rather than predicting absolute consumption.")
P("<b>In scope:</b> commercial and multifamily records from NYC LL84 (2022 to 2024), compared with EPA ENERGY STAR national reference values; baseline comparison; an interactive application with ranks and contributing factors. "
  "<b>Out of scope:</b> interval meter data, equipment-level fault detection, consumption forecasting, root-cause diagnosis, custom weather normalization and records with missing required fields.")

# ---------------------------------------------------------------- 2 data
P("2. Data", H1)
P("2.1 Sources", H2)
P(f"<b>Primary:</b> NYC Building Energy and Water Data Disclosure for Local Law 84 (2023 to Present; data for calendar year 2022 to present), downloaded from NYC Open Data on October 4, 2026. "
  f"The file contains {f0(b['raw_rows'])} property-year records ({f0(b['raw_by_year']['2022'])} in 2022, {f0(b['raw_by_year']['2023'])} in 2023 and {f0(b['raw_by_year']['2024'])} in 2024) across 265 columns. "
  f"<b>Secondary:</b> the EPA ENERGY STAR Portfolio Manager technical reference, <i>U.S. Energy Use Intensity by Property Type</i> (August 2024). Each LL84 record already carries the national median site EUI for its reference property type, "
  f"so that column is used as the peer benchmark feature, and the PDF table served as a cross-check. Austin's ECAD dataset, considered in the proposal, no longer publishes building-level records and was dropped.")
P("2.2 Cleaning and record classes", H2)
P(f"{f0(b['duplicate_property_years_removed'])} duplicate Property ID by year rows were resolved by keeping the most complete record (ties broken by latest submission date), leaving {f0(b['raw_rows'] - b['duplicate_property_years_removed'])} property-years. "
  f"Floor area is derived as site energy divided by site EUI, because the \"calculated GFA\" columns are about 99% empty. The derived value matches the reported largest-use floor area closely (median ratio 1.0002). "
  f"Every record is then assigned to one of three classes. Following the proposal, implausible records are kept as their own <b>data-quality anomaly</b> class instead of being discarded.")
sc = allr.groupby(["year", "status"]).size().unstack(fill_value=0)
rows = [["Year", "Model-ready", "Data-quality anomaly", "Out of scope"]] + [[str(y), f0(sc.loc[y, "model_ready"]), f0(sc.loc[y, "dq_anomaly"]), f0(sc.loc[y, "out_of_scope"])] for y in sc.index]
rows.append(["Total", f0(sc.model_ready.sum()), f0(sc.dq_anomaly.sum()), f0(sc.out_of_scope.sum())])
table(rows, [1.2, 1.4, 1.8, 1.4])
P(f"Data-quality rules: site EUI outside 5 to 1,000 kBtu/ft<super>2</super>, derived floor area below 25,000 ft<super>2</super> (the LL84 coverage threshold) and an invalid year built. "
  f"Out-of-scope records are mostly missing energy data, followed by property types with fewer than 50 buildings, parking structures (EUI per square foot is not meaningful) and test or design-stage records. "
  f"The model-ready set has {f0(b['model_ready_buildings'])} unique buildings across {b['property_types_in_model']} property types, which is more than the 60,000 to 80,000 property-years estimated in the proposal.")
P("2.3 Known data limitations", H2)
miss = allr.groupby("year").site_eui.apply(lambda s: s.isna().mean())
bullets([f"<b>Self-reporting errors and defaults.</b> {pc1(allr.default_values.mean())} of records carry the default-values flag and {pc1(allr.estimated_energy.mean())} the estimated-energy flag; {pc1((allr.occupancy == 100).mean())} report exactly 100% occupancy.",
         f"<b>Missing site EUI is growing.</b> {pc1(miss.loc[2022])} of 2022 records, {pc1(miss.loc[2023])} of 2023 and {pc1(miss.loc[2024])} of 2024 have no site EUI, so the 2024 hold-out may be a self-selected subset.",
         "<b>Operating hours are mostly unavailable.</b> Weekly operating hours exist only for office, medical office, financial office and restaurant uses, so the field is about 85% missing and is dropped from the global model by the 40% missingness rule.",
         "<b>Selection bias.</b> LL84 covers buildings over 25,000 ft<super>2</super>, and non-compliant owners are absent.",
         "<b>Annual granularity.</b> Only year-round anomalies can be detected, not seasonal or short-term faults.",
         f"<b>Multifamily dominates.</b> Multifamily Housing is {pc((ready.prop_type == 'Multifamily Housing').mean())} of model-ready records, so overall metrics mostly describe that class."])

# ---------------------------------------------------------------- 3 EDA
P("3. Exploratory Data Analysis", H1)
top = ready.prop_type.value_counts().head(8).index
rows = [["Property type", "Records", "Median EUI", "IQR", "National median"]]
for tname in top:
    s = ready[ready.prop_type == tname]
    rows.append([tname, f0(len(s)), f"{s.site_eui.median():.1f}", f"{s.site_eui.quantile(.25):.0f} to {s.site_eui.quantile(.75):.0f}", (f"{s.nat_median_site_eui.median():.1f}" if s.nat_median_site_eui.notna().any() else "n/a")])
P("Site EUI is right-skewed and varies strongly by property type, so the target is modeled on the natural-log scale. The table compares the NYC median with the EPA national median for each of the eight largest types (kBtu/ft<super>2</super>).")
table(rows, [2.3, 0.9, 1.0, 1.2, 1.2])
fig("03_eui_by_type_box.png", "Figure 1. Site EUI by property type (log scale), ten most common types.", max_h=3.4)
fig("04_eui_vs_gfa.png", "Figure 2. Site EUI against floor area with binned median trend, by property type.", max_h=3.6)
fig("09_missingness_by_type.png", "Figure 3. Share of records missing key fields by property type. Weekly operating hours are missing for most types.", max_h=3.6)
P(f"Other EDA findings: the ENERGY STAR score is strongly negatively related to EUI (correlation about -0.69), which is expected because the score is computed from energy use. It is therefore excluded from the model features to avoid leakage and is used only to define proxy labels. "
  f"{pc((ready.eui_yoy_ratio.dropna() > 3).mean() + (ready.eui_yoy_ratio.dropna() < 1/3).mean())} of consecutive-year pairs show an EUI change larger than 3x in either direction, which marks candidate entry errors.")

# ---------------------------------------------------------------- 4 methods
P("4. Methodology", H1)
P("4.1 Features and preprocessing", H2)
P("The model uses property type and borough (categorical), log floor area, building age, number of buildings, occupancy, log national median EUI for the reference property type, five fuel-mix shares (electricity, natural gas, district steam, fuel oil, other), and the default-value and estimated-energy flags. "
  "Fuel shares are computed from reported energy by fuel; an unreported fuel counts as zero. Missing benchmark values are imputed with the training-set property-type median and flagged. "
  "Source EUI, site energy, ENERGY STAR score and greenhouse-gas fields are not features because they are computed from the same energy data as the target.")
P("4.2 Data partitioning", H2)
P(f"Models are trained and tuned on 2022 to 2023 ({f0(w['n_train'])} records, {f0(w['train_buildings'])} buildings) and evaluated once on the held-out 2024 records ({f0(w['n_test'])} records). "
  f"Cross-validation folds are grouped by Property ID so the same building never appears on both sides of a fold. About {pc(w['test_buildings_also_in_train_pct'])} of 2024 buildings also appear in the training years, so the hold-out is temporal rather than building-level, "
  f"which mirrors how the model would be retrained and applied to a new reporting year.")
P("4.3 Models", H2)
bullets(["<b>Rule-based baseline.</b> Flag buildings above the 90th percentile of site EUI within their property type (proposal baseline). Because proxy anomalies include unusually low EUI, a two-sided version (outside the 10th to 90th percentile) is also reported.",
         "<b>LightGBM residual model (primary).</b> Gradient-boosted regression of log site EUI. The anomaly score is the residual (actual minus expected log EUI) standardized <i>per property type</i> with a robust median and MAD scale learned from out-of-fold training residuals and shrunk toward the global scale for small types.",
         f"<b>Isolation Forest (secondary).</b> Fit on the full feature set plus log EUI with contamination fixed at 10%. The tree count ({u['iforest_trees_chosen']}) is the smallest that stabilizes scores across five random seeds (mean Spearman at least 0.99, top-10% overlap at least 0.85).",
         "<b>One-Class SVM (comparison).</b> RBF kernel on a scaled 8,000-record training subset."])
P("4.4 Hyperparameter tuning", H2)
bp = w["lgbm_best_params"]
P(f"LightGBM was tuned with a 40-candidate randomized search and 5-fold cross-validation grouped by Property ID, scored by RMSE on log EUI. The best configuration had a cross-validated RMSE of {f3(w['lgbm_cv_rmse_log'])} "
  f"({bp['model__n_estimators']} trees, learning rate {bp['model__learning_rate']:.3f}, max depth {bp['model__max_depth']}, {bp['model__num_leaves']} leaves, subsample {bp['model__subsample']:.2f}, "
  f"column sample {bp['model__colsample_bytree']:.2f}, minimum {bp['model__min_child_samples']} samples per leaf). Cross-validated log RMSE across the {len(w['lgbm_search_candidates'])} tried configurations ranged from {f3(min(w['lgbm_search_candidates'].values()))} to {f3(max(w['lgbm_search_candidates'].values()))}.")
fig("15_flag_rate_per_type_scaling.png", "Figure 4. A single global residual scale over-flags property types with naturally wider residuals; per-type scaling evens out flag rates across the ten largest types (top 10% flagged overall).", max_h=3.5)
P(f"The spread (standard deviation) of flag rates across the ten largest types fell from {f3(m['flag_rate_spread_sd']['global_sigma'])} with one global scale to {f3(m['flag_rate_spread_sd']['per_type_sigma'])} with per-type scaling.")

# ---------------------------------------------------------------- 5 results
P("5. Evaluation and Results", H1)
P("There are no clean ground-truth anomaly labels, so evaluation uses four layers: regression accuracy, a proxy-label benchmark, stability and agreement, and a manual domain audit.")
P("5.1 Regression accuracy (2024 hold-out)", H2)
rows = [["Model", "RMSE (ln EUI)", "R-squared (ln EUI)", "MAPE (EUI)", "RMSE (kBtu/ft2)"]]
for k, n in [("property_type_median", "Property-type median"), ("linear_regression", "Linear regression"), ("lightgbm", "LightGBM (tuned)")]:
    r = reg[k]; rows.append([n, f3(r["rmse_log"]), f2(r["r2_log"]), pc(r["mape_eui"]), f"{r['rmse_eui']:.1f}"])
table(rows, [2.2, 1.1, 1.3, 1.0, 1.2])
fig("12_lgbm_actual_vs_expected.png", "Figure 5. Actual versus expected site EUI on the 2024 hold-out (left) and the distribution of standardized residuals, the anomaly score (right).", max_h=3.0)
P(f"LightGBM explains {pc(t['r2_log'])} of log-EUI variance, a clear gain over the property-type median and linear regression, but a large share of variation remains unexplained. This is expected with annual data, no operating-hours field and no ENERGY STAR score as an input, and it is why the output is framed as a prioritization signal and not a verdict.")
P("5.2 Proxy-label benchmark", H2)
pool_n, cnt = m["proxy_label_counts"]["pool"], m["proxy_label_counts"]
P(f"The benchmark pool is the {f0(pool_n)} scorable 2024 records, including data-quality anomalies. Proxy positives are records that fail plausibility screens ({f0(cnt['plausibility_fail'])}), have a year-over-year EUI shift above 3x ({f0(cnt['yoy_shift_gt3x'])}) "
  f"or have an extreme ENERGY STAR score of 1 to 5 or 95 to 100 ({f0(cnt['extreme_es_score'])}). {f0(cnt['any_proxy'])} records ({pc(cnt['any_proxy'] / pool_n)}) are positive under at least one definition. "
  f"The table reports PR-AUC and precision and recall among the top 10% of scores.")
def bench_table(key):
    rows = [["Method", "PR-AUC", "Precision@5%", "Precision@10%", "Recall@10%"]]
    for r in m["benchmark"][key]: rows.append([NAMES[r["method"]], f2(r["pr_auc"]), f2(r["precision@5%"]), f2(r["precision@10%"]), f2(r["recall@10%"])])
    return rows
P(f"<b>All proxy labels</b> (base rate {pc(cnt['any_proxy'] / pool_n)})", BODY); table(bench_table("any_proxy"), [3.0, 0.7, 0.95, 0.95, 0.8])
P(f"<b>Excluding plausibility failures</b> (base rate {pc(cnt['any_excl_plausibility'] / pool_n)})", BODY); table(bench_table("any_excl_plausibility"), [3.0, 0.7, 0.95, 0.95, 0.8])
fig("16_proxy_benchmark.png", "Figure 6. Proxy-label benchmark. Dashed line is the base rate (random ranking).", max_h=3.2)
bm = {r["method"]: r for r in m["benchmark"]["any_proxy"]}
P(f"<b>Interpretation.</b> The residual model clearly beats the one-sided P90 rule from the proposal (PR-AUC {f2(bm['LightGBM |resid| (per-type sigma)']['pr_auc'])} versus {f2(bm['P90 rule (EUI / type P90)']['pr_auc'])}), "
  f"but a two-sided percentile rule scores {f2(bm['P10/P90 rule (two-sided)']['pr_auc'])}, slightly higher than LightGBM. This should not be over-read. The proxy labels are functions of EUI itself (the ENERGY STAR score and the plausibility screens are derived from it), "
  f"so they reward any method that ranks raw EUI extremes. They cannot measure what the residual model is designed to do: identify buildings that are unusual <i>given their own characteristics</i> (for example a hospital that is unremarkable in raw EUI but high for its size and fuel mix) and explain why. "
  f"The Isolation Forest performs worst among the multivariate methods because it flags unusual combinations of features rather than EUI that departs from peers, which is not the anomaly definition used here.")
P("5.3 Stability and inter-model agreement", H2)
s1, s2 = m["stability_lightgbm"], m["stability_iforest"]
P(f"Across five random seeds, LightGBM rankings have a mean pairwise Spearman correlation of {f3(s1['mean_spearman'])} and a mean top-10% Jaccard overlap of {f2(s1['mean_top10_jaccard'])}; "
  f"for the Isolation Forest the figures are {f3(s2['mean_spearman'])} and {f2(s2['mean_top10_jaccard'])}. Rankings are stable overall, with more movement at the margin of the top 10%. "
  f"Agreement between methods is low to moderate (Figure 7), which shows that the methods surface different buildings and supports using explanations and expert review, not a single score, before acting on a flag.")
fig("17_model_agreement.png", "Figure 7. Rank correlation and top-10% overlap between methods on model-ready 2024 buildings.", max_h=3.0)
P("5.4 Explainability", H2)
P("SHAP values (exact TreeSHAP values from LightGBM) show what drives the <i>expected</i> EUI. The national median EUI of the building's reference type is the strongest driver, followed by electricity share, property type, building age and natural gas share. "
  "Higher electricity share lowers expected site EUI (electric end uses convert to site energy one to one, while on-site combustion adds fuel energy), and older buildings are expected to use more. "
  "The application shows a local SHAP waterfall for any selected building, so a reviewer can see why the expectation is what it is before judging the gap.")
fig("19_shap_global_importance.png", "Figure 8. Global SHAP feature importance (mean absolute effect on log EUI, 3,000 sampled 2024 buildings).", width=4.6, max_h=3.2)
fig("18_shap_summary.png", "Figure 9. SHAP summary: each point is a building; color shows the feature value.", width=4.8, max_h=4.2)
P("5.5 Domain expert audit", H2)
if audit:
    c = audit["counts"]
    P("A stratified, blind sample of 50 model-ready 2024 buildings (25 from the top 2% of anomaly scores, 25 from the middle of the ranking) was reviewed and classified as Plausible Anomaly, Data Error or Normal, using domain knowledge of building energy performance, with the model's rank hidden during review.")
    rows = [["Model rank group", "Plausible Anomaly", "Data Error", "Normal", "Not Normal"]]
    for g, n in [("top", "Top 2% (flagged)"), ("mid", "Middle of ranking")]:
        rows.append([n, c[g]["Plausible Anomaly"], c[g]["Data Error"], c[g]["Normal"], f"{c[g]['Plausible Anomaly'] + c[g]['Data Error']} of {audit['n_' + g]}"])
    table(rows, [1.9, 1.2, 1.0, 0.9, 1.2])
    fig("20_audit_confusion.png", "Figure 10. Audit results by model rank group and precision confusion matrix.", max_h=2.8)
    sig = "a statistically meaningful difference" if audit["fisher_exact_p_top_vs_mid_non_normal"] < 0.05 else "a difference that is not statistically distinguishable at this sample size"
    P(f"Among top-ranked buildings, {pc(audit['precision_top_any_non_normal'])} were judged anomalous (plausible anomaly or data error; {pc(audit['precision_top_plausible_anomaly'])} plausible anomalies and {pc(audit['data_error_share_top'])} data errors), "
      f"compared with {pc(audit['mid_any_non_normal'])} of mid-ranked buildings. A Fisher exact test gives {('p < 0.001' if audit['fisher_exact_p_top_vs_mid_non_normal'] < 0.001 else 'p = %.3f' % audit['fisher_exact_p_top_vs_mid_non_normal'])}, {sig}. With only 50 buildings and a single reviewer, these percentages are imprecise. Most flagged buildings were judged data errors, so the model works mainly as a screen for questionable reports, with a smaller share of true operational anomalies.")
else:
    P("<b>[AUDIT PENDING] This section must be completed before submission.</b> The blind audit workbook (reports/expert_audit_50_buildings.xlsx) has not been classified yet. "
      "After classifying all 50 buildings, run <i>python scripts/05_audit_results.py</i> and then <i>python scripts/06_build_report.py</i>; this section is then filled in automatically with precision for top-ranked versus mid-ranked buildings, a confusion matrix and a significance test.", PEND)
P("5.6 Model selection", H2)
P("LightGBM residual scoring with per-type scaling is the primary model. It is the only method that conditions on building characteristics, gives a continuous ranking, is stable across seeds and explains each flag with SHAP. "
  "The unsupervised models are retained as comparisons. The proposal states that final selection prioritizes expert-judged precision and explainability, so the audit result in Section 5.5 is the deciding evidence" + ("." if audit else " and is still pending."))

# ---------------------------------------------------------------- 6 application
P("6. Interactive Application", H1)
P(f"The model is delivered as a multipage Streamlit application (live application: {cfg['app_url']}; repository: {cfg['repo_url']}). The pages follow the course template:")
bullets(["<b>Biographical homepage</b> with a professional summary, headshot and links.", "<b>Resume page</b> with education, experience, an interactive career timeline and skills chart, and a PDF download.",
         "<b>Projects page</b> with portfolio cards, including this project.",
         "<b>Specific project page</b> with four modules: <i>Problem and Approach</i> (non-technical explanation); <i>Results and Performance</i> (model comparison, benchmark, stability, SHAP); "
         "<i>Explore Portfolio Rankings</i> (sidebar filters for property type, floor area, borough, year built, dominant fuel, score cutoff and algorithm; actual-versus-expected scatter plot; drill-down with peer comparison and a local SHAP waterfall; CSV export); "
         "and <i>Score Your Own Building</i> (a form that computes EUI from entered energy use and returns expected EUI, anomaly score, flags from three methods and a SHAP explanation)."])
P("The serialized pipelines, pre-scored 2024 table and figures are loaded with Streamlit caching. Automated smoke tests confirm that every page renders, the scoring form runs, empty filter selections are handled, and scoring a stored building through the form path reproduces its stored score exactly.")

# ---------------------------------------------------------------- 7 limitations
P("7. Limitations and Next Steps", H1)
bullets(["<b>Proxy-label circularity.</b> Proxy labels are derived from EUI, so they favor EUI-extreme rules. Expert-labeled audit data is the better evaluation and could also train a semi-supervised model.",
         "<b>Temporal, not building-level, hold-out.</b> Most 2024 buildings were seen in training years; a building-disjoint evaluation would be stricter.",
         f"<b>2024 reporting gaps.</b> {pc1(miss.loc[2024])} of 2024 records lack EUI, which may bias the hold-out toward buildings that report on time.",
         "<b>Missing operating hours.</b> A separate office-only model could use operating hours where they exist.",
         "<b>Thresholds.</b> The top-10% flag rate is a convention, not an optimized cutoff.",
         "<b>Scope of conclusions.</b> A flag indicates a building that merits review. The cause may be inefficiency, an unusual use or a reporting error. The model does not diagnose root causes.",
         "<b>Future work.</b> Cross-reference flagged buildings with Local Law 33 letter grades and Local Law 97 compliance data; use interval meter data and custom weather normalization; retrain on other municipal disclosure datasets; calibrate thresholds with audit labels."])

# ---------------------------------------------------------------- 8 AI + reproducibility
P("8. AI Usage Statement", H1)
P("In line with the AI usage plan in the approved proposal, AI tools (Claude, by Anthropic) were used as coding, writing and debugging assistants. Claude drafted the data pipeline, exploratory notebook, model training and evaluation scripts, "
  "the Streamlit application code and the first draft of this report from the approved proposal, and all code was executed locally and checked (results reproduce on re-run, and tests cover the application). "
  "Scope, data source, feature choices, evaluation design and the domain audit follow the approved proposal and the author's engineering judgment. An inline attribution log is maintained in the repository (docs/AI_USAGE_LOG.md).")
P("9. Reproducibility", H1)
rows = [["Step", "Command", "Output"], ["1", "python scripts/01_build_dataset.py", "Cleaned and classified data, build log"], ["2", "python scripts/make_eda_notebook.py (then run notebook)", "EDA notebook and figures"],
        ["3", "python scripts/02_train_baseline_and_lightgbm.py", "Baseline, tuned LightGBM, 2024 metrics"], ["4", "python scripts/03_isolation_forest_ocsvm.py", "Isolation Forest, One-Class SVM"],
        ["5", "python scripts/04_evaluate_and_package.py", "Per-type scaling, benchmark, SHAP, audit workbook, app files"], ["6", "python scripts/05_audit_results.py", "Audit precision and confusion matrix"],
        ["7", "python scripts/06_build_report.py", "This report"], ["8", "python tests/test_app.py", "Application smoke tests"]]
table(rows, [0.5, 3.4, 2.6])

# ---------------------------------------------------------------- references
P("References", H1)
refs = ["EPA ENERGY STAR. (2024, August). <i>U.S. energy use intensity by property type</i> (Technical Reference). U.S. Environmental Protection Agency.",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &amp; Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. <i>Advances in Neural Information Processing Systems, 30</i>.",
        "Liu, F. T., Ting, K. M., &amp; Zhou, Z.-H. (2008). Isolation forest. <i>2008 Eighth IEEE International Conference on Data Mining</i>, 413-422.",
        "Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>Advances in Neural Information Processing Systems, 30</i>.",
        "Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., &amp; Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. <i>Nature Machine Intelligence, 2</i>, 56-67.",
        "New York City Department of Buildings. (2026). <i>NYC building energy and water data disclosure for Local Law 84 (2023 to present)</i> [Data set]. NYC Open Data. Downloaded October 4, 2026.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825-2830.",
        "Scholkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., &amp; Williamson, R. C. (2001). Estimating the support of a high-dimensional distribution. <i>Neural Computation, 13</i>(7), 1443-1471."]
for r in refs: story.append(Paragraph(r, ParagraphStyle("ref", parent=BODY, leftIndent=18, firstLineIndent=-18, fontSize=9, leading=12)))

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(inch, 0.55 * inch, "Building Energy Benchmarking Anomaly Detection | DTSC 691"); canvas.drawRightString(letter[0] - inch, 0.55 * inch, f"Page {doc.page}"); canvas.restoreState()
out = DOCS / "Final_Project_Overview.pdf"
SimpleDocTemplate(str(out), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                  title="Building Energy Benchmarking Anomaly Detection: Project Overview", author=cfg["student"]).build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("wrote", out)
