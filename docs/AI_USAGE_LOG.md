# AI Usage Log

Project: DTSC 691 capstone, Anomaly Detection in NYC Building Energy Benchmarking Data (LL84)
Student: Julie Bounds | Mentor: Jasmeet Bhatia

## Tool used
Claude (Anthropic), used as a coding and writing assistant through the Claude desktop app. Code was written by Claude, then run and tested on my laptop.

## What Claude did
| Area | Claude's contribution | How it was checked |
|---|---|---|
| Data cleaning (`src/cleaning.py`, script 01) | Drafted dedup, derived fields, row classification | Row counts logged in `reports/dataset_build_log.json`; I reviewed the exclusion rules |
| EDA notebook | Generated notebook and figures | Figures reviewed by me |
| Baselines and LightGBM (script 02) | Drafted pipeline, grouped CV, randomized search | Run locally; metrics in `reports/week4_metrics.json` |
| Residual scaling, Isolation Forest, One-Class SVM (script 03) | Drafted per-type scaling, tree-count stability test | Seed stability and Jaccard results logged |
| Proxy benchmark and packaging (script 04) | Drafted metrics, figures, audit workbook, app data | Outputs in `reports/final_metrics.json` |
| Streamlit app (`app/`) | Drafted pages, tabs, scoring form | Smoke tests in `tests/test_app.py` |
| Final report (script 06) | Drafted text and tables; numbers pulled from JSON files | I review the PDF before submission |

## What I did myself
- Defined the project, scope, and proposal.
- Classify the 50-building blind audit (this is the real validation, so it must be my judgment).
- Write my own bio, resume content, and project descriptions in the app.
- Create the GitHub repo, deploy the app, and record the video.

## Known limitations of AI-assisted work
- The proxy labels are partly circular with the models; the report says so.
- Claude cannot verify results beyond what ran locally, so I review all numbers against the JSON logs.
