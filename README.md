# Forecasting the Effective Federal Funds Rate

This project began as a team project during AI4ALL in summer 2025. The first version explored whether macroeconomic indicators and several machine-learning models could forecast the Effective Federal Funds Rate. In 2026, I returned to it and rebuilt the analysis so that every forecast uses only past information and the results can be reproduced from the data in this repository.

The current version is a one-month-ahead benchmark comparing persistence, Ridge regression, and Random Forest. I developed the model design on data through February 2017, fixed the choices, and then evaluated them on a second Federal Reserve snapshot covering March 2017 through August 2026.

**Main finding:** Ridge has the lowest external-period RMSE, at **0.1606 percentage points** versus **0.1922** for persistence. Its MAE advantage is much smaller, and the uncertainty interval includes zero. Random Forest remains worse than both simpler approaches.

![EFFR forecasts on the external period](reports/external/forecast.png)

## External validation

All features and hyperparameters come from the earlier benchmark. The models are refitted on every usable observation through February 2017, then evaluated on 114 untouched monthly observations. Each prediction uses the rate observed in the previous month; no later-period values are used for fitting or model selection.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Persistence | 0.1922 | 0.0996 | 0.9894 |
| Ridge | **0.1606** | **0.0934** | **0.9926** |
| Random Forest | 0.2368 | 0.1393 | 0.9839 |

RMSE and MAE are percentage points. For scale, Ridge's RMSE is about 16.1 basis points and persistence's is about 19.2 basis points.

A paired moving-block bootstrap preserves short runs of neighboring months when estimating uncertainty. Ridge improves RMSE over persistence by **0.0317 percentage points** (95% interval **0.0070 to 0.0568**). Its MAE improvement is **0.0062** (95% interval **−0.0174 to 0.0335**), so that smaller difference is not clearly distinguishable from zero. Full results are in [metrics.csv](reports/external/metrics.csv) and [uncertainty.csv](reports/external/uncertainty.csv).

## Reproduce the results

Use Python **3.12**. Both official-source data snapshots are included, so the run does not need an API key or network access.

```bash
git clone https://github.com/agalkova/effr-forecasting-ml.git
cd effr-forecasting-ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q

effr-train --data data/effr_monthly.csv --output reports
effr-external \
  --development-data data/effr_monthly.csv \
  --external-data data/effr_external.csv \
  --frozen-design reports/run.json \
  --output reports/external
effr-verify \
  --data data/effr_monthly.csv \
  --expected reports \
  --external-data data/effr_external.csv \
  --external-expected reports/external
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The package commands can also be run as Python modules. Direct dependencies are listed in `pyproject.toml`; `requirements-lock.txt` records the tested environment.

## Forecast design

For target month *t*, the inputs are EFFR lags at 1, 2, 3, 6, and 12 months; trailing means and standard deviations over 3, 6, and 12 months; and the preceding monthly change. Every input ends at *t−1*.

The development benchmark uses a chronological 80/20 split and five expanding training folds. Scaling for Ridge is fitted inside each fold. Training CV selected persistence and fixed the Ridge penalty and Random Forest settings before the later data were evaluated.

![Development-period forecasts on the chronological holdout](reports/forecast.png)

The holdout also shows why Random Forest was a poor choice for this problem: it could not extrapolate below the rate floor in its training data during the post-2008 near-zero-rate period. The external period contains a different mix of rising, near-zero, and falling rates, and Ridge handles the larger monthly moves better.

See [methodology.md](docs/methodology.md) for the forecast timing, development results, leakage audit, and uncertainty procedure. Source details and file hashes are in [data/README.md](data/README.md).

## Outputs

- `reports/` contains the development split's predictions, model-selection results, feature importance, figures, and run metadata.
- `reports/external/` contains dated external predictions, metrics, uncertainty intervals, the final figure, and run metadata.
- `tests/` checks feature timing, split isolation, data continuity, frozen parameters, dated outputs, and deterministic uncertainty estimates.
- GitHub Actions reruns the tests and both evaluations, then compares regenerated artifacts with the committed results.

## What this project predicts

EFFR is closely guided by the Federal Reserve's target rate or range, which makes persistence a meaningful benchmark. This project tests short-run predictability in the realized monthly rate. It uses a rolling one-step evaluation: after a month is observed, that value becomes available for forecasting the next month.

The project does not try to anticipate FOMC decisions. That would require meeting dates, the target range, market expectations, and real-time macroeconomic vintages available on each forecast date.

## Project history and license

The original exploratory project was developed by **Anastasia Galkova and Bakr Bouhaya** during AI4ALL in summer 2025. Anastasia later rebuilt the data pipeline, evaluation, tests, and external validation presented here. See [LICENSE](LICENSE) for code licensing and [data/README.md](data/README.md) for data attribution and terms.
