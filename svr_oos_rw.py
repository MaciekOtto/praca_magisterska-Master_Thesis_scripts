"""
svr_oos_rw.py - Prognozy out-of-sample: SVR

Dla każdej spółki trenuje model SVR (Support Vector Regression, kernel
RBF) do prognozowania zmienności (realized variance, r²) na podstawie
cech opóźnionych (LAG=5) w schemacie rozszerzającego się okna
(TRAIN_SIZE=1250, model przetrenowywany okresowo co RETRAIN_EVERY
kroków). Obliczenia zrównoleglone, z checkpointami.

Wejście: dane1000stopy.xlsx
Wyjście: svr_prognozy_oos.parquet, svr_rmse_mae.xlsx,
         checkpoints_svr/

----------------------------------------------------------------------

svr_oos_rw.py - Out-of-sample forecasts: SVR

For each company, trains an SVR (Support Vector Regression, RBF
kernel) model to forecast volatility (realized variance, r²) using
lagged features (LAG=5) in an expanding-window scheme (TRAIN_SIZE=1250,
the model is periodically retrained every RETRAIN_EVERY steps).
Computation is parallelized, with checkpoints.

Input: dane1000stopy.xlsx
Output: svr_prognozy_oos.parquet, svr_rmse_mae.xlsx,
        checkpoints_svr/
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import multiprocessing as mp
import time
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ── Ustawienia ────────────────────────────────────────────────────────────────

INPUT_FILE       = 'dane1000stopy.xlsx'
CHECKPOINT_DIR   = 'checkpoints_svr'
OUT_PARQUET      = 'svr_prognozy_oos.parquet'
OUT_EXCEL        = 'svr_rmse_mae.xlsx'

TRAIN_SIZE       = 1250
LAG              = 5
ROLLING_STD_W    = 20
RETRAIN_EVERY    = 25    

SVR_PARAMS = dict(
    kernel  = 'rbf',
    C       = 1.0,
    epsilon = 0.01,
    gamma   = 'scale',
)

N_WORKERS        = max(1, mp.cpu_count() - 1)
CHECKPOINT_EVERY = 50

# ── Budowanie features ─

def build_features(returns_arr):
    r  = returns_arr.astype(float)
    r2 = r ** 2
    n  = len(r)

    rows = []
    start = ROLLING_STD_W + LAG

    for t in range(start, n - 1):
        row = {}
        for k in range(1, LAG + 1):
            row[f'r2_lag_{k}']  = r2[t - k]
            row[f'ret_lag_{k}'] = r[t - k]
        row['rolling_std_20'] = float(np.std(r[t - ROLLING_STD_W:t]))
        row['target'] = r2[t + 1]
        row['t_idx']  = t
        rows.append(row)

    if not rows:
        return None, None, None

    df_f = pd.DataFrame(rows).dropna()
    feat_cols = [c for c in df_f.columns if c not in ('target', 't_idx')]
    return df_f[feat_cols].values, df_f['target'].values, df_f['t_idx'].values


# ── Jedna spółka ──────────────────────────────────────────────────────────────

def process_ticker(ticker, returns_arr, train_size, retrain_every, svr_params):
    empty = {
        'ticker':    ticker,
        'forecasts': np.array([]),
        'metrics':   {'Spółka': ticker, 'RMSE': np.nan,
                      'MAE': np.nan, 'N_valid': 0, 'Model': 'SVR'},
    }

    X, y, t_idx = build_features(returns_arr)
    if X is None:
        return empty

    test_mask  = t_idx >= train_size
    train_mask = ~test_mask

    if train_mask.sum() < 50 or test_mask.sum() < 5:
        return empty

    X_test = X[test_mask]
    y_test = y[test_mask]
    n_test = len(X_test)

    forecasts  = np.full(n_test, np.nan)
    last_model = None
    last_scaler = None

    for step in range(n_test):
        if step % retrain_every == 0:
            cut  = train_mask.sum() + step
            X_tr = X[:cut]
            y_tr = y[:cut]

            # Skalowanie 
            scaler  = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)

            model = SVR(**svr_params)
            model.fit(X_tr_sc, y_tr)
            last_model  = model
            last_scaler = scaler

        if last_model is not None:
            X_step_sc   = last_scaler.transform(X_test[step:step+1])
            forecasts[step] = float(last_model.predict(X_step_sc)[0])

    valid = ~np.isnan(forecasts)
    if valid.sum() > 5:
        rmse = float(np.sqrt(mean_squared_error(y_test[valid], forecasts[valid])))
        mae  = float(mean_absolute_error(y_test[valid], forecasts[valid]))
    else:
        rmse = mae = np.nan

    return {
        'ticker':    ticker,
        'forecasts': forecasts,
        'metrics':   {'Spółka': ticker, 'RMSE': rmse,
                      'MAE': mae, 'N_valid': int(valid.sum()), 'Model': 'SVR'},
    }


# ── Batch ───

def process_batch(args):
    (batch_idx, tickers_batch, data_dict,
     train_size, retrain_every, svr_params,
     checkpoint_dir, checkpoint_every) = args

    os.makedirs(checkpoint_dir, exist_ok=True)
    results = []

    for i, ticker in enumerate(tickers_batch):
        res = process_ticker(ticker, data_dict[ticker],
                             train_size, retrain_every, svr_params)
        results.append(res)

        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(tickers_batch):
            cp = os.path.join(checkpoint_dir,
                              f'batch_{batch_idx:03d}_step_{i:04d}.pkl')
            pd.to_pickle(results, cp)

    return results


# ── Helpers ───

def load_data():
    print(f"Wczytanie danych z: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    date_col = next((c for c in df.columns
                     if 'data' in c.lower() or 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df.dropna(inplace=True)
    return df


def load_completed(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return set(), []
    done, results = set(), []
    for cp in sorted(glob.glob(os.path.join(checkpoint_dir, '*.pkl'))):
        try:
            for r in pd.read_pickle(cp):
                if r['ticker'] not in done:
                    done.add(r['ticker'])
                    results.append(r)
        except Exception:
            pass
    return done, results


def save_outputs(all_results):
    metrics = pd.DataFrame([r['metrics'] for r in all_results])
    fc_dict = {r['ticker']: r['forecasts'] for r in all_results
               if len(r['forecasts']) > 0}
    pd.DataFrame(fc_dict).to_parquet(OUT_PARQUET)
    print(f"Prognozy: {OUT_PARQUET}  "
          f"({os.path.getsize(OUT_PARQUET)/1e6:.1f} MB)")

    summary = pd.DataFrame([{
        'Model':        'SVR',
        'RMSE_mediana': metrics['RMSE'].median(),
        'RMSE_srednia': metrics['RMSE'].mean(),
        'RMSE_std':     metrics['RMSE'].std(),
        'MAE_mediana':  metrics['MAE'].median(),
        'MAE_srednia':  metrics['MAE'].mean(),
        'N_spółek_OK':  metrics['RMSE'].notna().sum(),
    }])

    with pd.ExcelWriter(OUT_EXCEL, engine='openpyxl') as writer:
        metrics.to_excel(writer, sheet_name='Surowe',       index=False)
        summary.to_excel(writer, sheet_name='Podsumowanie', index=False)

    print(f"Metryki: {OUT_EXCEL}")
    print(f"\n── SVR Podsumowanie ──")
    print(f"RMSE mediana: {metrics['RMSE'].median():.8f}")
    print(f"MAE  mediana: {metrics['MAE'].median():.8f}")
    print(f"Spółek OK:    {metrics['RMSE'].notna().sum()}")


# ── MAIN ──

if __name__ == '__main__':
    t0 = time.time()

    df = load_data()
    tickers   = df.columns.tolist()
    TEST_SIZE = len(df) - TRAIN_SIZE

    print(f"Spółek: {len(tickers)} | Train: {TRAIN_SIZE} | Test: {TEST_SIZE}")
    print(f"Retrenowanie co {RETRAIN_EVERY} dni | "
          f"Kernel: {SVR_PARAMS['kernel']} | Workery: {N_WORKERS}")

    done, existing = load_completed(CHECKPOINT_DIR)
    remaining = [t for t in tickers if t not in done]

    if done:
        print(f"Checkpointy: {len(done)} gotowych, {len(remaining)} pozostało")

    if not remaining:
        print("Wszystkie spółki gotowe...")
        save_outputs(existing)
        exit()

    data_dict  = {t: df[t].values for t in remaining}
    batches    = [list(b) for b in np.array_split(remaining, N_WORKERS) if len(b)]
    batch_args = [
        (i, batch, data_dict, TRAIN_SIZE, RETRAIN_EVERY,
         SVR_PARAMS, CHECKPOINT_DIR, CHECKPOINT_EVERY)
        for i, batch in enumerate(batches)
    ]

    print(f"\nStart SVR rolling... (~3-5h)\n")
    with mp.Pool(processes=N_WORKERS) as pool:
        results_list = pool.map(process_batch, batch_args)

    all_results = existing + [r for batch in results_list for r in batch]
    save_outputs(all_results)

    print(f"\nCzas całkowity: {(time.time()-t0)/3600:.2f} h")
    print("GOTOWE.")
