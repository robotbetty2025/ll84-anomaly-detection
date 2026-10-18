# Building Energy Benchmarking Anomaly Detection (NYC LL84)

Live app: https://julie-bounds-energy-anomaly.streamlit.app | Code: https://github.com/robotbetty2025/ll84-anomaly-detection | Video: https://drive.google.com/file/d/1jFocvouM3ryzQk8X3ai6gbT5n8o4kIlC/view?usp=sharing

DTSC 691 capstone. Ranks commercial and multifamily buildings by how far their site EUI departs from what
comparable buildings are expected to use, instead of using a static percentile cutoff.

## Status (Oct 18, 2026)

Done: data pipeline, EDA, rule-based baselines, tuned LightGBM residual model with per-type scaling, Isolation Forest, One-Class SVM, proxy-label benchmark, seed stability, SHAP, 50-building blind expert audit, deployed 4-page Streamlit app, and the final report (`docs/Final_Project_Overview.pdf`). See `docs/AI_USAGE_LOG.md` for how AI assistance was used.

## Data
* NYC LL84 benchmarking disclosures, calendar years 2022 to 2024 (103,259 property-year rows, 265 columns).
  Source: NYC Open Data, "NYC Building Energy and Water Data Disclosure for Local Law 84 (2023 to Present)".
* EPA ENERGY STAR "U.S. Energy Use Intensity by Property Type" national median table (Aug 2024).
  The LL84 file already carries each building's national median site EUI, so that column is used as the reference feature.
* Put the CSV in `data/raw/` (or leave it in the parent capstone folder). Override with the `LL84_CSV` environment variable.

## How to run
```
pip install -r requirements.txt
python scripts/01_build_dataset.py              # clean, classify, engineer features -> data/processed/
python scripts/make_eda_notebook.py             # (re)generate notebooks/01_eda.ipynb, then run it in Jupyter
python scripts/02_train_baseline_and_lightgbm.py   # baseline + LightGBM (resumable; re-run until it finishes)
python scripts/03_isolation_forest_ocsvm.py        # Isolation Forest, One-Class SVM, seed stability
python scripts/04_evaluate_and_package.py          # per-type scaling, proxy benchmark, SHAP, audit workbook, app artifacts
python scripts/05_audit_results.py                  # after you finish the blind audit workbook
python scripts/06_build_report.py                  # builds docs/Final_Project_Overview.pdf
python tests/test_app.py                           # smoke-test the Streamlit app
streamlit run app/Home.py                          # run the app locally
```
Deploy: push this folder to a GitHub repo (the `.gitignore` already excludes the raw CSV and big generated data; the app only needs
`app/`, `src/`), then on Streamlit Community Cloud choose the repo, main file `app/Home.py`. `app/requirements.txt` is the slim dependency list.
Before sharing: edit `app/content.py` (bio, resume, projects, links), add `app/assets/headshot.png` and `app/assets/resume.pdf`, and set `PLACEHOLDERS_REMAIN = False`.
Run all commands from the project root. Step 2 caches tuning results in `reports/cv_search_cache.json`
and can be re-run if it stops partway.

## Layout
```
src/config.py       paths, thresholds, column mapping      src/cleaning.py   load, de-dup, classify records
src/features.py     features + per-type median imputer     src/baseline.py   P90-within-type rule
src/models.py       LightGBM pipeline                      src/anomaly.py    Isolation Forest, One-Class SVM
src/scoring.py      per-type residual scaling              src/evaluation.py proxy-label benchmark helpers
scripts/            runnable steps 01 to 04                notebooks/01_eda.ipynb   EDA
app/                Streamlit app (Home.py, pages/)        reports/          metrics, logs, figures, audit workbook
tests/              app smoke tests                        docs/             progress summaries
```

## Key decisions
* Target is site EUI (kBtu/ft2), modeled as ln(EUI). Gross floor area is derived as site energy / site EUI because the
  "calculated GFA" columns are about 99% empty (it matches the largest-use GFA, median ratio 1.0002).
* Records are classed `model_ready`, `dq_anomaly` (implausible EUI, GFA, or year built; kept as its own class) or `out_of_scope`.
* ENERGY STAR score, source EUI, site energy and GHG fields are NOT model features (they are computed from the target).
* Weekly operating hours exist only for four use types (about 85% missing overall), so the proposal's 40% missingness rule drops it.
* Anomaly score = LightGBM residual (actual minus expected ln EUI), scaled per property type with a robust median/MAD scale.
* Train and validate on 2022 to 2023 (CV grouped by Property ID), test on 2024.
* Parking is excluded (EUI per ft2 is not meaningful for parking structures).

## AI usage log
Claude (Anthropic) assisted with drafting the pipeline code and EDA notebook in this repository; all code was run and
checked locally, and modeling decisions are the author's. Add entries here as work continues.
