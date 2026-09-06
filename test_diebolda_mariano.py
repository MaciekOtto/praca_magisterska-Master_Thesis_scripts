"""
test_diebolda_mariano.py - Test Diebolda-Mariano (GARCH vs. ML)

Porównuje trafność prognoz out-of-sample par modeli GARCH-rodziny
(GARCH, GJR-GARCH) i modeli uczenia maszynowego (RF, LSTM, SVR) za
pomocą testu Diebolda-Mariano opartego na błędzie kwadratowym względem
realized variance. Test przeprowadzany jest osobno dla każdej spółki
(firm-by-firm), a wyniki agregowane są przekrojowo (m.in. odsetek
spółek, dla których dany model istotnie przewyższa drugi).

Wejście: dane1000stopy.xlsx, garch_prognozy_oos_fixed.parquet,
         rf_prognozy_oos.parquet, lstm_prognozy_oos.parquet,
         svr_prognozy_oos.parquet
Wyjście: wyniki_diebold_mariano.xlsx

----------------------------------------------------------------------

test_diebolda_mariano.py - Diebold-Mariano test (GARCH vs. ML)

Compares out-of-sample forecast accuracy for pairs of GARCH-family
models (GARCH, GJR-GARCH) and machine learning models (RF, LSTM, SVR)
using the Diebold-Mariano test based on squared-error loss relative to
realized variance. The test is run separately for each company
(firm-by-firm), and the results are aggregated cross-sectionally
(e.g., the share of companies where one model significantly
outperforms the other).

Input: dane1000stopy.xlsx, garch_prognozy_oos_fixed.parquet,
       rf_prognozy_oos.parquet, lstm_prognozy_oos.parquet,
       svr_prognozy_oos.parquet
Output: wyniki_diebold_mariano.xlsx
"""

import re
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

RETURNS_FILE = "dane1000stopy.xlsx"
TEST_SIZE = 250  
ALPHA = 0.05

GARCH_FAMILY_FILE = "garch_prognozy_oos_fixed.parquet"
GARCH_FAMILY_MODELS = ["GARCH", "EGARCH", "GJR-GARCH", "APARCH"]

ML_FORECAST_FILES = {
    "RF": "rf_prognozy_oos.parquet",
    "LSTM": "lstm_prognozy_oos.parquet",
    "SVR": "svr_prognozy_oos.parquet",
}

MODEL_PAIRS = [
    ("GARCH", "RF"),
    ("GARCH", "LSTM"),
    ("GARCH", "SVR"),
    ("GJR-GARCH", "RF"),
    ("GJR-GARCH", "LSTM"),
    ("GJR-GARCH", "SVR"),
]

OUTPUT_FILE = "wyniki_diebold_mariano.xlsx"

_QUOTED = re.compile(r"'([^']+)'")

# ============================================================================


def load_realized_variance(returns_file: str, test_size: int) -> pd.DataFrame:
    df = pd.read_excel(returns_file, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df.columns = [c.replace("_Close", "") for c in df.columns]
    rv = df.iloc[-test_size:] ** 2
    return rv


def _parse_garch_column(col_label) -> tuple | None:
    if isinstance(col_label, tuple):
        if len(col_label) >= 2:
            return str(col_label[0]), str(col_label[1])
        return None

    s = str(col_label)
    matches = _QUOTED.findall(s)
    if len(matches) >= 2:
        return matches[0], matches[1]
    return None


def _find_date_column(columns) -> object | None:
    for c in columns:
        if isinstance(c, tuple) and len(c) >= 1 and str(c[0]) == "Data":
            return c
        matches = _QUOTED.findall(str(c))
        if len(matches) == 1 and matches[0] == "Data":
            return c
        if str(c) in ("Data", "data", "index", "Index"):
            return c
    return None


def load_garch_family(path: str, tickers: list, test_size: int) -> dict:
    raw = pd.read_parquet(path)

    date_col = _find_date_column(raw.columns)
    if date_col is not None:
        raw = raw.set_index(date_col)
    else:
        print("UWAGA: nie znaleziono kolumny z datą w pliku GARCH - "
              "sprawdź strukturę pliku.")

    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[raw.index.notna()]
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]

    print("Zakres dat GARCH:", raw.index.min(), raw.index.max())

    by_model = {}
    unparsed = []
    for c in raw.columns:
        if c == "__index_level_0__":
            continue
        parsed = _parse_garch_column(c)
        if parsed is None:
            unparsed.append(c)
            continue
        model, ticker = parsed
        ticker = ticker.replace("_Close", "")
        by_model.setdefault(model, {})[ticker] = raw[c]

    if unparsed:
        print(f"UWAGA: {len(unparsed)} kolumn nie udało się sparsować "
              f"(pominięte): {unparsed[:5]}{'...' if len(unparsed) > 5 else ''}")

    models_found = sorted(by_model.keys())
    print("Modele znalezione:", models_found)

    out = {}
    for model in models_found:
        if model not in GARCH_FAMILY_MODELS:
            continue
        sub = pd.DataFrame(by_model[model])
        sub = sub.iloc[-test_size:]
        common = [t for t in sub.columns if t in tickers]
        out[model] = sub[common]

    return out

def load_forecast_wide(path: str, tickers: list) -> pd.DataFrame:
    raw = pd.read_parquet(path)

    cols_lower = [str(c).lower() for c in raw.columns]
    is_long = {"ticker", "forecast"}.issubset(set(cols_lower)) or \
              {"symbol", "forecast"}.issubset(set(cols_lower))

    if is_long:
        rename_map = {}
        for c in raw.columns:
            cl = str(c).lower()
            if cl in ("ticker", "symbol"):
                rename_map[c] = "ticker"
            elif cl in ("date", "data"):
                rename_map[c] = "date"
            elif cl in ("forecast", "prediction", "pred", "yhat"):
                rename_map[c] = "forecast"
        raw = raw.rename(columns=rename_map)
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        wide = raw.pivot(index="date", columns="ticker", values="forecast")
        wide = wide.sort_index()
        wide = wide[~wide.index.duplicated(keep="last")]
    else:
        date_col = _find_date_column(raw.columns)
        if isinstance(raw.index, pd.DatetimeIndex):
            wide = raw.sort_index()
        elif date_col is not None:
            raw = raw.set_index(date_col)
            raw.index = pd.to_datetime(raw.index, errors="coerce")
            raw = raw[raw.index.notna()]
            wide = raw.sort_index()
        else:
            wide = raw

        wide.columns = [str(c).replace("_Close", "").replace("_forecast", "") for c in wide.columns]

    common = [t for t in wide.columns if t in tickers]
    return wide[common]


def align_ml_forecast(wide: pd.DataFrame, rv_index: pd.DatetimeIndex, test_size: int) -> pd.DataFrame:

    if isinstance(wide.index, pd.DatetimeIndex):
        return wide.iloc[-test_size:]

    n = len(wide)
    if n > len(rv_index):
        raise ValueError(
            f"Plik ML ma więcej wierszy ({n}) niż realized variance "
            f"({len(rv_index)}) - dopasowanie pozycyjne się nie zgadza."
        )
    aligned = wide.copy()
    aligned.index = rv_index[-n:]
    return aligned


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, loss: str = "squared"):
    if loss == "squared":
        l1 = e1 ** 2
        l2 = e2 ** 2
    elif loss == "absolute":
        l1 = np.abs(e1)
        l2 = np.abs(e2)
    else:
        raise ValueError("loss musi być 'squared' lub 'absolute'")

    d = l1 - l2
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    if d_var == 0 or np.isnan(d_var):
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(d_var / n)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))

    return dm_stat, p_value


def main():
    rv = load_realized_variance(RETURNS_FILE, TEST_SIZE)
    rv = rv[~rv.index.duplicated(keep="last")]
    tickers = list(rv.columns)
    print(f"Wczytano realized variance dla {len(tickers)} spółek, "
          f"{rv.shape[0]} obserwacji testowych.")

    forecasts = {}

    if Path(GARCH_FAMILY_FILE).exists():
        garch_dict = load_garch_family(GARCH_FAMILY_FILE, tickers, TEST_SIZE)
        for model, df in garch_dict.items():
            forecasts[model] = df
            print(f"Wczytano prognozy dla modelu {model}: "
                  f"{df.shape[1]} spółek, {df.shape[0]} obs.")
    else:
        print(f"UWAGA: plik {GARCH_FAMILY_FILE} nie istnieje - "
              f"modele GARCH-family pominięte.")

    needed_ml = {m for pair in MODEL_PAIRS for m in pair if m in ML_FORECAST_FILES}
    for model in sorted(needed_ml):
        path = ML_FORECAST_FILES[model]
        if not Path(path).exists():
            print(f"UWAGA: plik {path} nie istnieje - pomijam model {model}")
            continue
        wide = load_forecast_wide(path, tickers)
        wide = align_ml_forecast(wide, rv.index, TEST_SIZE)
        forecasts[model] = wide
        print(f"Wczytano prognozy dla modelu {model}: "
              f"{forecasts[model].shape[1]} spółek, {forecasts[model].shape[0]} obs. "
              f"(zakres dat: {wide.index.min()} - {wide.index.max()})")

    results = []

    for model_1, model_2 in MODEL_PAIRS:
        if model_1 not in forecasts or model_2 not in forecasts:
            print(f"Pomijam parę {model_1} vs {model_2} - brak danych.")
            continue

        f1 = forecasts[model_1]
        f2 = forecasts[model_2]

        common_tickers = sorted(set(rv.columns) & set(f1.columns) & set(f2.columns))

        n_significant_1_better = 0
        n_significant_2_better = 0
        n_not_significant = 0
        n_skipped = 0

        for ticker in common_tickers:
            actual = rv[ticker].values
            s1 = f1[ticker]
            s2 = f2[ticker]

            s1 = s1[~s1.index.duplicated(keep="last")]
            s2 = s2[~s2.index.duplicated(keep="last")]

            pred1 = s1.reindex(rv.index).values
            pred2 = s2.reindex(rv.index).values

            mask = ~(np.isnan(actual) | np.isnan(pred1) | np.isnan(pred2))
            if mask.sum() < 30:
                n_skipped += 1
                continue

            e1 = pred1[mask] - actual[mask]
            e2 = pred2[mask] - actual[mask]

            dm_stat, p_value = diebold_mariano(e1, e2, loss="squared")

            if np.isnan(p_value):
                n_skipped += 1
                continue

            results.append({
                "model_1": model_1,
                "model_2": model_2,
                "ticker": ticker,
                "dm_stat": dm_stat,
                "p_value": p_value,
                "n_obs": mask.sum(),
            })

            if p_value < ALPHA:
                if dm_stat < 0:
                    n_significant_1_better += 1
                else:
                    n_significant_2_better += 1
            else:
                n_not_significant += 1

        total = n_significant_1_better + n_significant_2_better + n_not_significant
        print(f"\n{model_1} vs {model_2} (n={total}, pominięto={n_skipped}):")
        if total:
            print(f"  {model_1} istotnie lepszy: {n_significant_1_better} "
                  f"({100*n_significant_1_better/total:.1f}%)")
            print(f"  {model_2} istotnie lepszy: {n_significant_2_better} "
                  f"({100*n_significant_2_better/total:.1f}%)")
            print(f"  brak istotnej różnicy:    {n_not_significant} "
                  f"({100*n_not_significant/total:.1f}%)")
        else:
            print("  brak danych")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        with pd.ExcelWriter(OUTPUT_FILE) as writer:
            results_df.to_excel(writer, sheet_name="szczegoly", index=False)

            summary_rows = []
            for (m1, m2), grp in results_df.groupby(["model_1", "model_2"]):
                n = len(grp)
                sig_1 = ((grp["p_value"] < ALPHA) & (grp["dm_stat"] < 0)).sum()
                sig_2 = ((grp["p_value"] < ALPHA) & (grp["dm_stat"] > 0)).sum()
                not_sig = n - sig_1 - sig_2
                summary_rows.append({
                    "model_1": m1,
                    "model_2": m2,
                    "n_spolek": n,
                    f"{m1}_istotnie_lepszy": sig_1,
                    f"{m1}_istotnie_lepszy_%": round(100 * sig_1 / n, 1),
                    f"{m2}_istotnie_lepszy": sig_2,
                    f"{m2}_istotnie_lepszy_%": round(100 * sig_2 / n, 1),
                    "brak_istotnej_roznicy": not_sig,
                    "brak_istotnej_roznicy_%": round(100 * not_sig / n, 1),
                })
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="podsumowanie", index=False)

        print(f"\nZapisano wyniki do: {OUTPUT_FILE}")
    else:
        print("\nNie uzyskano żadnych wyników - sprawdź pliki wejściowe.")


if __name__ == "__main__":
    main()
