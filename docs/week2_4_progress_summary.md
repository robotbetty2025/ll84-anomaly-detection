# Weeks 2 to 4 catch-up: progress summary (Oct 4, 2026)

## Data (matches the approved proposal)
* LL84 file: 103,259 rows (30,485 in 2022; 33,684 in 2023; 39,090 in 2024). 265 columns, not 481.
* After removing 775 duplicate Property ID/year rows: 102,484 property-years.
* Model-ready: {'2022': 26956, '2023': 28077, '2024': 28184} (total 83,217 across 36,088 buildings, 43 property types). Another 4,999 records are held as a data-quality anomaly class.
* Multifamily is about 70% of model-ready records. Site EUI is missing for 5.2% of 2022, 8.2% of 2023 and 21.8% of 2024 records.

## Models (trained 2022-2023, tested on 2024)
| Model | RMSE (log EUI) | R2 (log EUI) | MAPE (EUI) |
|---|---|---|---|
| Property-type median | 0.580 | 0.09 | 58% |
| Linear regression | 0.459 | 0.43 | 38% |
| LightGBM (12-candidate search, grouped 5-fold CV) | 0.399 | 0.57 | 30% |

* Rule-based P90 baseline flags 9.0% of 2024 buildings. LightGBM top-10% flags overlap with it at Jaccard 0.36 (Spearman rank correlation 0.64), so the two methods disagree on many buildings.
* Top LightGBM drivers by gain: national median EUI, electricity share, property type, building age.

## Things to know before the next steps
1. Residual spread differs a lot by property type (std of standardized residual about 0.85 for multifamily vs about 2.1 for manufacturing), so one global sigma over-flags small types (for example 27% of manufacturing plants). Next step: per-type robust sigma.
2. About 82% of 2024 buildings also appear in 2022-2023, so the 2024 hold-out is temporal, not a building-level split.
3. 2024 has far more missing EUI values than earlier years, which may bias the hold-out set.
4. Operating hours could not be used globally (85% missing), which weakens the proposal's operating-hours driver.
