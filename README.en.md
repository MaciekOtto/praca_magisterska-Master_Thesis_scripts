# GARCH Models vs. Machine Learning Methods for Volatility Forecasting — Master's Thesis Scripts

This repository contains the Python source code used in the empirical part of a master's thesis comparing classical GARCH-family models (GARCH, EGARCH, GJR-GARCH, APARCH) against machine learning methods (Random Forest, LSTM, SVR) for forecasting return volatility, in the context of the weak-form Efficient Market Hypothesis (EMH) and the Adaptive Market Hypothesis (AMH).

The empirical sample covers **1,000 NASDAQ-listed companies**, using daily closing prices from **2020-02-12 to 2025-11-12**, sourced via the `yfinance` library.

## Table of Contents

- [Project Goal](#project-goal)
- [Repository Structure](#repository-structure)
- [Pipeline (Run Order)](#pipeline-run-order)
- [Requirements](#requirements)
- [Input Data](#input-data)
- [Known Limitations / TODO](#known-limitations--todo)

## Project Goal

The thesis primarily tests whether return volatility is predictable at all (null hypothesis: volatility is completely unpredictable) and whether machine learning methods systematically outperform classical GARCH models in out-of-sample forecast accuracy. Additionally, using CAPM and the Fama-French three-factor model (FF3), the project examines whether a firm's exposure to market, size (SMB), and value (HML) risk factors explains cross-sectional differences in forecast errors.

## Repository Structure

The repository currently has a flat structure (all scripts in the root directory). Files are grouped below by function:

| Group | Files | Description |
|---|---|---|
| **Data collection** | `stock_scraping.py` | Downloads daily closing prices for tickers listed in `nasdaq_top500.csv` via `yfinance`, filters out companies with incomplete history, saves `dane1000close.xlsx` |
| **Data preparation** | `log_returns.py` | Computes log returns from `dane1000close.xlsx` → `dane1000stopy.xlsx` |
| **Preliminary tests** | `test_adf.py`, `ADF_vis.py`, `test_arch.py`, `arch_vis.py`, `test_wilska.py`, `desc_stats_and_vis.py` | ADF test (stationarity), ARCH-effect test (on ARIMA(1,0,1) residuals), Shapiro-Wilk normality test, descriptive statistics of returns with visualizations |
| **GARCH-family estimation (full sample)** | `garch.py`, `egarch.py`, `gjrgarch.py`, `aparch.py` | Parameter estimation of GARCH(1,1), EGARCH, GJR-GARCH, and APARCH models per firm, plus histograms of parameter distributions |
| **Residual diagnostics** | `rozklad_garch4testy.py`, `rozklad_egarch4testy.py`, `rozklad_gjr4testy.py`, `rozklad_aparch4testy.py` | Statistical tests on the standardized residuals of the fitted GARCH-family models |
| **Out-of-sample forecasting (expanding window, firm-by-firm)** | `OOS_rolling_windowGARCH.py`, `RF_oos_rw.py`, `lstm_oos_rw.py`, `svr_oos_rw.py` | Expanding-window volatility forecasts (`TRAIN_SIZE=1250`, 250-observation test window) for all six models; outputs saved as `.parquet` (forecasts) and `.xlsx` (RMSE/MAE); checkpoints saved periodically |
| **Data format fix** | `fix_garchparquet.py` | Fixes MultiIndex column names in `garch_prognozy_oos.parquet` (serialization issue with `np.str_`) |
| **Diebold-Mariano test** | `test_diebolda_mariano.py` | Compares forecast accuracy of GARCH vs. ML model pairs (squared-error loss) → `wyniki_diebold_mariano.xlsx` |
| **Forecast error comparison** | `error_vis.py` | Comparative RMSE/MAE plots across all six models |
| **CAPM / FF3 cross-sectional regression** | `capm_3factors.py`, `3factors_R2_visuals.py`, `3factors_visuals_scatter.py` | Two-stage regression: (1) firm-level FF3/CAPM beta estimation, (2) cross-sectional regression of RMSE on betas; visualizations of R² and RMSE-beta relationships |

## Pipeline (Run Order)

1. **`stock_scraping.py`** — downloads price data (requires `nasdaq_top500.csv` in the root directory) → `dane1000close.xlsx`
2. **`log_returns.py`** — computes log returns → `dane1000stopy.xlsx`
3. **Preliminary tests** (optional, any order): `test_adf.py` → `ADF_vis.py`; `test_arch.py` → `arch_vis.py`; `test_wilska.py`; `desc_stats_and_vis.py`
4. **Full-sample GARCH-family estimation**: `garch.py`, `egarch.py`, `gjrgarch.py`, `aparch.py`, followed by the corresponding `rozklad_*4testy.py` scripts
5. **Out-of-sample forecasting**: `OOS_rolling_windowGARCH.py` and `RF_oos_rw.py`, `ltsm_oos_rw.py`, `svr_oos_rw.py` (can be run independently/in parallel — each writes its own checkpoints and output files)
6. If you get an error related to the GARCH forecast parquet file's column format → run **`fix_garchparquet.py`**
7. **`test_diebolda_mariano.py`** — requires the `.parquet` files from step 5 (for the GARCH family, the `_fixed` version — see step 6)
8. **`error_vis.py`** — requires the `*_rmse_mae.xlsx` files from step 5
9. **`capm_3factors.py`** — requires `dane1000stopy.xlsx`, the Fama-French factor file `F-F_Research_Data_Factors_daily.txt` (see *Input Data*), and the `*_rmse_mae.xlsx` files
10. **`3factors_R2_visuals.py`** and **`3factors_visuals_scatter.py`** — ⚠️ currently require manually extracting specific sheets from step 9's output file into separate intermediate files (`ff3_regresja_przekrojowa_2.xlsx`, `ff3_regresja_przekrojowa_vis.xlsx`) — see *Known Limitations*.

## Requirements

Scripts tested with Python 3.11.0. Key dependencies (full pinned list in `requirements.txt`):

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
arch
scikit-learn
tensorflow
yfinance
pyarrow
openpyxl
```

It's recommended to use a dedicated virtual environment and install via:

```bash
pip install -r requirements.txt
```

## Input Data

- `nasdaq_top500.csv` — the input ticker list, included in this repository.
- `dane1000close.xlsx`, `dane1000stopy.xlsx` — generated locally by `stock_scraping.py` and `log_returns.py`; **not included in the repository** (size / data pulled live from Yahoo Finance).
- `F-F_Research_Data_Factors_daily.txt` — daily Fama-French factors (Mkt-RF, SMB, HML, RF), downloaded from [Kenneth French's data library (Dartmouth)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html);

## Known Limitations / TODO

- Input/output file paths are hardcoded in the scripts (no `argparse` or config file).
- The `*_blankvisuals.py` files are earlier/alternate versions of the main GARCH-family estimation scripts (without the diagnostics section) — kept in the repository for reference, pending cleanup.
- Pipeline step 10 (`3factors_R2_visuals.py`, `3factors_visuals_scatter.py`) currently requires manually splitting out specific sheets from `ff3_regresja_przekrojowa.xlsx` into separate files — automating this is planned.

