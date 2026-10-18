# Video Script and Shot List (target 20-22 minutes)

Tip: record in 5 short clips and stitch them. Keep the app open in a browser and the report PDF open beside it.

## 0:00-3:00 Problem framing (show: report title page, then Figure 1 or slide)
- Say: NYC Local Law 84 makes large buildings report energy use every year. Some reports are wrong (typos, unit errors) and some buildings really are wasteful. Both are worth finding.
- Say: no one has labels for which buildings are "anomalies," so this is an unsupervised problem.
- Say: my goal is a ranked list a reviewer could use, and a tool that explains why a building was flagged.
- Say: the data is 2022 to 2024 LL84 data; I train on 2022-2023 and test on 2024.

## 3:00-8:00 Data pipeline and EDA (show: report section 2-3, EDA figures, repo `src/cleaning.py`)
- Walk the cleaning steps: duplicates, plausibility range for EUI, 25,000 ft2 minimum, parking excluded.
- Show row counts from the dataset build log.
- Show 2 to 3 EDA figures (EUI by property type, skew, missingness).
- Mention what was dropped (the sparse fields) and why.
- Mention the ENERGY STAR leakage concern and how I handled it.

## 8:00-14:00 Modeling and evaluation (show: report section 4-5, figures 15-19)
- Baselines: property-type median, linear regression, then LightGBM (R2 0.584 on 2024).
- Explain residual scoring: a building is unusual if it uses far more or less than similar buildings predict.
- Explain per-type scaling so one property type does not dominate the flags.
- Isolation Forest and One-Class SVM as the other approach; mention the 800-tree stability choice.
- Proxy benchmark table: be honest that the simple P10/P90 rule scored about as well (PR-AUC 0.65 vs 0.62), and that the proxy labels are partly circular.
- Show SHAP chart and the blind audit result (fill in once the audit is done).

## 14:00-20:00 Live app demo (show: Streamlit app)
1. Home page (30 sec).
2. Anomaly Detector, Problem and Approach tab (1 min).
3. Results and Performance tab (1.5 min).
4. Explore Portfolio Rankings: apply filters, open a flagged building, read the explanation (2 min).
5. Score Your Own Building: enter a made-up building, show the score and SHAP drivers (1.5 min).

## 20:00-22:00 Limitations and next steps
- No ground truth; proxy labels circular; temporal hold-out only.
- Missing fields (fuel mix, occupancy) limit accuracy.
- Next: more years, weather normalization, a reviewer feedback loop to build real labels.
- Thank mentor Jasmeet Bhatia.
