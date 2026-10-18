# DTSC 691 progress summary: Week 6 prep (Oct 4, 2026)

Everything below was run end to end and the results reproduce. Code, figures, models and the app are in
`DTSC 691 Capstone/ll84-anomaly-detection/` (see README.md there).

## What was done
1. **LightGBM tuning finished.** 40-candidate randomized search, 5-fold CV grouped by Property ID. Best CV RMSE (ln EUI) 0.406.
   2024 hold-out: RMSE 0.393, R2 0.58, MAPE 29% (property-type median R2 0.09, linear regression R2 0.43). Gains over the earlier 12-candidate search were small, so the model has largely plateaued with these features.
2. **Per-type residual scaling.** Robust (median/MAD) center and scale per property type, shrunk toward the global value for small types. The spread of flag rates across the 10 biggest types fell from SD 0.080 to 0.028.
3. **Isolation Forest + One-Class SVM.** Isolation Forest uses 800 trees (smallest count with mean seed Spearman >= 0.99 and top-10% overlap >= 0.85). The One-Class SVM is fit on an 8,000-row subset.
4. **Proxy-label benchmark** on 29,916 scorable 2024 records (incl. data-quality anomalies). Proxy positives: plausibility failures 1,775, YoY EUI shifts over 3x 816, extreme ENERGY STAR scores 4,791.
5. **Stability.** Across 5 seeds: LightGBM mean Spearman 0.971 (top-10% Jaccard 0.89); Isolation Forest 0.997 (0.87).
6. **SHAP.** Global importance and summary plot (exact TreeSHAP from LightGBM). Top drivers: national median EUI, electricity share, property type, building age, gas share.
7. **Audit workbook ready for you:** `reports/expert_audit_50_buildings.xlsx`: 50 buildings (25 top-ranked, 25 mid-ranked), shuffled and blind, with a dropdown (Plausible Anomaly / Data Error / Normal). The ranking key is on the last sheet, so open that only after you finish. This is your domain-judgment step; I did not classify any building.
8. **Streamlit app built and smoke-tested** (4 pages, 4 project modules, scoring form, SHAP waterfalls). Automated tests confirm every page renders, the form works, and scoring a stored building through the form path reproduces its stored score exactly.

## Proxy benchmark (P@10% = precision in the top 10% of scores, R@10% = recall)
| Method | PR-AUC (all) | P@10% (all) | R@10% (all) | PR-AUC (excl. plausibility) | P@10% (excl.) | R@10% (excl.) |
|---|---|---|---|---|---|---|
| P90 rule (EUI / type P90) | 0.30 | 0.46 | 0.21 | 0.21 | 0.36 | 0.20 |
| P10/P90 rule (two-sided) | 0.65 | 0.84 | 0.38 | 0.53 | 0.72 | 0.40 |
| LightGBM abs residual (global sigma) | 0.58 | 0.67 | 0.30 | 0.42 | 0.50 | 0.28 |
| LightGBM abs residual (per-type sigma) | 0.62 | 0.74 | 0.34 | 0.49 | 0.61 | 0.34 |
| Isolation Forest | 0.32 | 0.36 | 0.16 | 0.24 | 0.26 | 0.14 |
| One-Class SVM | 0.45 | 0.57 | 0.26 | 0.25 | 0.32 | 0.18 |

## Important findings to be aware of
* **The ML model does not clearly beat a fair rule-based baseline on the proxy labels.** The proposal's rule (above P90 only) is clearly beaten by LightGBM, but that rule is one-sided, while the proxies include unusually LOW EUI. A two-sided P10/P90 rule does about as well as (or slightly better than) LightGBM. The proxies are built from EUI itself (ENERGY STAR score and plausibility screens are functions of EUI), so they reward raw-EUI extremes and cannot measure the residual model's real strength: unusual *given the building's characteristics*, with SHAP explanations. The 50-building audit is the fair test. This should be stated honestly in the report.
* **Isolation Forest is among the weakest methods** on these proxies and agrees little with LightGBM (rank correlation about 0.16). It flags unusual combinations of features, not EUI departing from peers, which is not quite your anomaly definition.
* **Model selection is provisional.** Your proposal says final selection prioritizes expert-judged precision and explainability. LightGBM residual scoring is the working choice because it is the only method that explains each flag, but decide after the audit.
* The hold-out is temporal, not building-level (about 82% of 2024 buildings appear in 2022-23), and 2024 has more missing EUI values (21.8% vs 5.2% in 2022).

## What is left for you (and me)
* **You:** complete the audit workbook; fill in `app/content.py` (bio, resume, links, projects), add `app/assets/headshot.png` and `resume.pdf`; create the GitHub repo and deploy (see README).
* **Then:** compute audit precision (top vs mid) and add a confusion matrix; write the final report and record the video. I can do those once the audit is back.
