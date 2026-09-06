"""
Regresja przekrojowa — czynniki Famy-Frencha (3-czynnikowy)
============================================================
Cel: wyjaśnić dlaczego jedne spółki są trudniejsze do prognozowania niż inne.

Kroki:
  1. Dla każdej spółki estymowane sa ekspozycje na czynniki FF3:
       r_it - RF_t = α_i + β_mkt_i*(Mkt-RF_t) + β_smb_i*SMB_t + β_hml_i*HML_t + ε_it
     → otrzymujemy β_mkt_i, β_smb_i, β_hml_i dla każdej spółki

  2. Regresja przekrojowa:
       RMSE_i = a + b1*β_mkt_i + b2*β_smb_i + b3*β_hml_i + u_i
     → czy ekspozycja na czynniki ryzyka wyjaśnia błąd prognozy?

  3. Wersja CAPM (tylko β_mkt) jako porównanie

Wejście:
  - dane1000stopy.xlsx              (stopy zwrotu spółek)
  - F-F_Research_Data_Factors_daily.txt 
  - garch_rmse_mae.xlsx             (RMSE z modeli — używamy GJR-GARCH jako benchmark)
  - rf_rmse_mae.xlsx, lstm_rmse_mae.xlsx, svr_rmse_mae.xlsx

Wyjście:
  - ff3_bety_spolek.xlsx            (β dla każdej spółki)
  - ff3_regresja_przekrojowa.xlsx   (wyniki regresji przekrojowej)


capm_3factors.py — CAPM / Fama-French (FF3) cross-sectional regression

Goal: test whether a firm's exposure to risk factors explains
cross-sectional differences in forecast errors (RMSE). Two stages:
(1) for each company, CAPM (market only) and FF3 (market, SMB, HML)
betas are estimated from its returns and the Fama-French factors;
(2) a cross-sectional regression of RMSE (from each of the six models)
on the estimated betas is run, with HAC standard errors.

Input: dane1000stopy.xlsx, F-F_Research_Data_Factors_daily.txt,
       garch_rmse_mae.xlsx, rf_rmse_mae.xlsx, lstm_rmse_mae.xlsx,
       svr_rmse_mae.xlsx
Output: ff3_bety_spolek.xlsx, ff3_regresja_przekrojowa.xlsx
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
import warnings
warnings.filterwarnings('ignore')

# ── Ustawienia ────────────────────────────────────────────────────────────────

RETURNS_FILE = 'dane1000stopy.xlsx'
FF_FILE      = 'F-F_Research_Data_Factors_daily.txt' 

# Pliki z RMSE 
RMSE_FILES = {
    'GARCH':     ('garch_rmse_mae.xlsx',  'GJR-GARCH'),  
    'RF':        ('rf_rmse_mae.xlsx',     None),
    'LSTM':      ('lstm_rmse_mae.xlsx',   None),
    'SVR':       ('svr_rmse_mae.xlsx',    None),
}

OUT_BETY   = 'ff3_bety_spolek.xlsx'
OUT_WYNIKI = 'ff3_regresja_przekrojowa.xlsx'

# ── Wczytanie czynników FF ────────────────────────────────────────────────────

def load_ff_factors(ff_file):
    import io
    
    with open(ff_file, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue
        first_word = clean_line.split()[0]
        if first_word.isdigit() and len(first_word) == 8:
            data_lines.append(clean_line)
    
    if not data_lines:
        raise ValueError(f"Nie znaleziono danych w pliku {ff_file}!")

    df = pd.DataFrame([l.split() for l in data_lines])
    
    df = df.iloc[:, :5] 
    df.columns = ['Date', 'Mkt_RF', 'SMB', 'HML', 'RF']
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    for col in ['Mkt_RF', 'SMB', 'HML', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100
    
    df.set_index('Date', inplace=True)
    df.dropna(inplace=True)
    
    print(f"Czynniki FF: {len(df)} obserwacji.")
    return df


# ── Wczytanie stóp zwrotu ─────────────────────────────────────────────────────

def load_returns(returns_file):
    df = pd.read_excel(returns_file)
    date_col = next((c for c in df.columns
                     if 'data' in c.lower() or 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.'), errors='coerce')

    n_missing = int(df.isna().sum().sum())
    print(f"Stopy zwrotu: {len(df.columns)} spółek, {len(df)} dni "
          f"(braki: {n_missing} komórek — obsłużone w regresji)")
    return df


# ── Wczytanie RMSE ────────────────────────────────────────────────────────────

def load_rmse_all(rmse_files):
  
    dfs = []
    for model_name, (filepath, col_name) in rmse_files.items():
        try:
            df_raw = pd.read_excel(filepath, sheet_name='Surowe')

            if col_name:
                df_raw = df_raw[df_raw['Model'] == col_name]

            df_raw = df_raw[['Spółka', 'RMSE']].copy()
            df_raw.columns = ['Spółka', model_name]
            df_raw.set_index('Spółka', inplace=True)
            dfs.append(df_raw)
            print(f"Wczytano RMSE: {model_name} ({len(df_raw)} spółek)")
        except Exception as e:
            print(f"Uwaga: nie mozna wczytać {filepath}: {e}")

    if not dfs:
        raise ValueError("Brak plików RMSE — sprawdź ścieżki.")

    df_rmse = pd.concat(dfs, axis=1)
    # Średnia RMSE po wszystkich modelach jako zmienna zależna (opcja)
    df_rmse['RMSE_srednia'] = df_rmse.mean(axis=1)
    return df_rmse


# ── Krok 1: Estymacja bet FF per spółka ──────────────────────────────────────

def estimate_betas(df_returns, df_ff):
    """
    Dla każdej spółki estymacja modelu FF3:
      r_it - RF_t = α + β_mkt*(Mkt-RF) + β_smb*SMB + β_hml*HML + ε

    """
    df_ff_aligned = df_ff.reindex(df_returns.index).ffill()

    df_ff_aligned = df_ff_aligned.dropna()

    common_idx = df_returns.index.intersection(df_ff_aligned.index)
    ret = df_returns.loc[common_idx]
    ff  = df_ff_aligned.loc[common_idx]

    print(f"\nEstymacja bet FF3 dla {len(ret.columns)} spółek "
          f"({len(common_idx)} wspólnych dni, po forward-fill FF)...")

    results = []
    for i, ticker in enumerate(ret.columns):
        r_excess = ret[ticker] - ff['RF']

        X = sm.add_constant(ff[['Mkt_RF', 'SMB', 'HML']])
        y = r_excess

        mask = ~(y.isna() | X.isna().any(axis=1))
        if mask.sum() < 100:
            results.append({'Spółka': ticker, 'beta_mkt': np.nan,
                            'beta_smb': np.nan, 'beta_hml': np.nan,
                            'alpha': np.nan, 'R2': np.nan})
            continue

        try:
            ols = sm.OLS(y[mask], X[mask]).fit(
                cov_type='HAC', cov_kwds={'maxlags': 5})
            results.append({
                'Spółka':    ticker,
                'alpha':     ols.params['const'],
                'beta_mkt':  ols.params['Mkt_RF'],
                'beta_smb':  ols.params['SMB'],
                'beta_hml':  ols.params['HML'],
                'R2':        ols.rsquared,
                'N_obs':     int(mask.sum()),
            })
        except Exception:
            results.append({'Spółka': ticker, 'beta_mkt': np.nan,
                            'beta_smb': np.nan, 'beta_hml': np.nan,
                            'alpha': np.nan, 'R2': np.nan})

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(ret.columns)}]")

    df_bety = pd.DataFrame(results).set_index('Spółka')
    print(f"Gotowe. Spółek z betami: {df_bety['beta_mkt'].notna().sum()}")
    return df_bety

# ── Krok 2: Regresja przekrojowa ─────────────────────────────────────────────

def cross_sectional_regression(df_bety, df_rmse, dep_var):
    """
    Regresja przekrojowa:
      RMSE_i = a + b1*β_mkt_i + b2*β_smb_i + b3*β_hml_i + u_i

    """
    # Łączymy bety z RMSE
    df = df_bety[['beta_mkt', 'beta_smb', 'beta_hml']].join(
        df_rmse[[dep_var]], how='inner').dropna()

    if len(df) < 50:
        print(f"Za mało obserwacji dla {dep_var}: {len(df)}")
        return None

    y = df[dep_var]

    # Model FF3
    X_ff3 = sm.add_constant(df[['beta_mkt', 'beta_smb', 'beta_hml']])
    ols_ff3 = sm.OLS(y, X_ff3).fit(cov_type='HC3')  

    # Model CAPM 
    X_capm = sm.add_constant(df[['beta_mkt']])
    ols_capm = sm.OLS(y, X_capm).fit(cov_type='HC3')

    return {
        'dep_var':   dep_var,
        'N':         len(df),
        'ff3':       ols_ff3,
        'capm':      ols_capm,
    }


def format_results(reg_results):
    rows = []
    for res in reg_results:
        if res is None:
            continue
        for model_type in ['ff3', 'capm']:
            ols = res[model_type]
            row = {
                'Zmienna zależna': res['dep_var'],
                'Model':           'FF3' if model_type == 'ff3' else 'CAPM',
                'N':               res['N'],
                'R²':              round(ols.rsquared, 4),
                'R² adj.':         round(ols.rsquared_adj, 4),
            }
            for param in ols.params.index:
                label = param.replace('beta_', 'β_').replace('const', 'stała')
                row[f'coef_{label}']   = round(ols.params[param], 6)
                row[f'pval_{label}']   = round(ols.pvalues[param], 4)
                row[f'tstat_{label}']  = round(ols.tvalues[param], 3)
            rows.append(row)
    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 1. Wczytanie danych
    df_ff      = load_ff_factors(FF_FILE)
    df_returns = load_returns(RETURNS_FILE)
    df_rmse    = load_rmse_all(RMSE_FILES)

    # 2. Estymacja bet FF3 per spółka
    df_bety = estimate_betas(df_returns, df_ff)

    # Zapis bet
    with pd.ExcelWriter(OUT_BETY, engine='openpyxl') as writer:
        df_bety.to_excel(writer, sheet_name='Bety_FF3')

        # Statystyki opisowe bet
        df_bety.describe().round(4).to_excel(writer, sheet_name='Statystyki_bet')
    print(f"\nBety zapisane: {OUT_BETY}")
    print("\nStatystyki bet (mediana):")
    print(df_bety[['beta_mkt', 'beta_smb', 'beta_hml', 'R2']].median().round(4))

    # 3. Regresja przekrojowa — dla każdego modelu osobno
    dep_vars = list(df_rmse.columns) 
    reg_results = []
    for dv in dep_vars:
        print(f"\nRegresja przekrojowa — zmienna zależna: {dv}")
        res = cross_sectional_regression(df_bety, df_rmse, dv)
        reg_results.append(res)
        if res:
            print(f"  FF3:  R²={res['ff3'].rsquared:.4f}  "
                  f"β_mkt p={res['ff3'].pvalues['beta_mkt']:.4f}  "
                  f"β_smb p={res['ff3'].pvalues['beta_smb']:.4f}  "
                  f"β_hml p={res['ff3'].pvalues['beta_hml']:.4f}")
            print(f"  CAPM: R²={res['capm'].rsquared:.4f}  "
                  f"β_mkt p={res['capm'].pvalues['beta_mkt']:.4f}")

    # 4. Zapis wyników
    df_tabela = format_results(reg_results)

    with pd.ExcelWriter(OUT_WYNIKI, engine='openpyxl') as writer:
        df_tabela.to_excel(writer, sheet_name='Regresja_przekrojowa', index=False)
        df_bety.join(df_rmse).to_excel(writer, sheet_name='Dane_do_regresji')

        for res in reg_results:
            if res is None:
                continue
            dv = res['dep_var'][:20]  
            summary_rows = []
            for model_type, label in [('ff3', 'FF3'), ('capm', 'CAPM')]:
                ols = res[model_type]
                for param in ols.params.index:
                    summary_rows.append({
                        'Model regresji': label,
                        'Zmienna':        param,
                        'Współczynnik':   round(ols.params[param], 6),
                        't-stat':         round(ols.tvalues[param], 3),
                        'p-value':        round(ols.pvalues[param], 4),
                        'Istotny 5%':     'TAK' if ols.pvalues[param] < 0.05 else 'NIE',
                    })
                summary_rows.append({
                    'Model regresji': label, 'Zmienna': '---',
                    'Współczynnik': None, 't-stat': None,
                    'p-value': None,
                    'Istotny 5%': f'R²={ols.rsquared:.4f} N={res["N"]}',
                })
            pd.DataFrame(summary_rows).to_excel(
                writer, sheet_name=f'Regr_{dv}', index=False)

    print(f"\nWyniki regresji przekrojowej zapisane: {OUT_WYNIKI}")
    print("\nGOTOWE.")
