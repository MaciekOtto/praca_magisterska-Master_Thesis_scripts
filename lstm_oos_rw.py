"""
lstm_oos_rw.py - Prognozy out-of-sample: LSTM

Dla każdej spółki trenuje jednowarstwową sieć LSTM (Keras/TensorFlow)
do prognozowania zmienności (realized variance, r²) w schemacie
rozszerzającego się okna (TRAIN_SIZE=1250, model przetrenowywany
okresowo co RETRAIN_EVERY kroków). Obliczenia z checkpointami co 50
spółek (CPU-only, zgodnie ze sprzętem użytym w pracy).

Wejście: dane1000stopy.xlsx
Wyjście: lstm_prognozy_oos.parquet, lstm_rmse_mae.xlsx,
         checkpoints_lstm/

----------------------------------------------------------------------

lstm_oos_rw.py - Out-of-sample forecasts: LSTM

For each company, trains a single-layer LSTM network (Keras/
TensorFlow) to forecast volatility (realized variance, r²) using an
expanding-window scheme (TRAIN_SIZE=1250, the model is periodically
retrained every RETRAIN_EVERY steps). Computation with checkpoints
every 50 companies (CPU-only, matching the hardware used for the
thesis).

Input: dane1000stopy.xlsx
Output: lstm_prognozy_oos.parquet, lstm_rmse_mae.xlsx,
        checkpoints_lstm/
"""

import pandas as pd
import numpy as np
import time
import os
import glob
import warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')

INPUT_FILE       = 'dane1000stopy.xlsx'
CHECKPOINT_DIR   = 'checkpoints_lstm'
OUT_PARQUET      = 'lstm_prognozy_oos.parquet'
OUT_EXCEL        = 'lstm_rmse_mae.xlsx'

TRAIN_SIZE       = 1250
SEQ_LEN          = 10
RETRAIN_EVERY    = 50    
CHECKPOINT_EVERY = 50

LSTM_UNITS  = 16        
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 0.001

N_WORKERS   = max(1, mp.cpu_count() - 1)

# ── Budowanie sekwencji ───────────────────────────────────────────────────────

def build_sequences(returns_arr):
    r    = returns_arr.astype(float)
    r2   = r ** 2
    absr = np.abs(r)
    n    = len(r)

    std20 = np.full(n, np.nan)
    for t in range(20, n):
        std20[t] = np.std(r[t-20:t])

    X_list, y_list, t_list = [], [], []
    for t in range(20 + SEQ_LEN, n - 1):
        seq = np.column_stack([
            r2[t - SEQ_LEN:t],
            absr[t - SEQ_LEN:t],
            std20[t - SEQ_LEN:t],
        ])
        if np.any(np.isnan(seq)):
            continue
        X_list.append(seq)
        y_list.append(r2[t + 1])
        t_list.append(t)

    if not X_list:
        return None, None, None

    return np.array(X_list), np.array(y_list), np.array(t_list)


# ── Jedna spółka ──────────────────────────────────────────────────────────────

def process_ticker(ticker, returns_arr, train_size, retrain_every):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['OMP_NUM_THREADS'] = '1'
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    def build_model():
        m = Sequential([
            Input(shape=(SEQ_LEN, 3)),
            LSTM(LSTM_UNITS, activation='tanh'),
            Dense(8, activation='relu'),
            Dense(1),
        ])
        m.compile(optimizer=Adam(LR), loss='mse')
        return m

    empty = {
        'ticker':    ticker,
        'forecasts': np.array([]),
        'metrics':   {'Spółka': ticker, 'RMSE': np.nan,
                      'MAE': np.nan, 'N_valid': 0},
    }

    X, y, t_idx = build_sequences(returns_arr)
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
    es = EarlyStopping(monitor='val_loss', patience=5,
                       restore_best_weights=True, verbose=0)

    for step in range(n_test):
        if step % retrain_every == 0:
            cut       = train_mask.sum() + step
            X_tr, y_tr = X[:cut], y[:cut]
            val_split = max(1, int(len(X_tr) * 0.1))

            model = build_model()
            model.fit(
                X_tr[:-val_split], y_tr[:-val_split],
                validation_data=(X_tr[-val_split:], y_tr[-val_split:]),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[es],
                verbose=0,
            )
            last_model = model
            tf.keras.backend.clear_session()
            last_model = model 

        if last_model is not None:
            forecasts[step] = float(
                last_model.predict(X_test[step:step+1], verbose=0)[0, 0])

    valid = ~np.isnan(forecasts)
    if valid.sum() > 5:
        rmse = float(np.sqrt(np.mean((forecasts[valid] - y_test[valid]) ** 2)))
        mae  = float(np.mean(np.abs(forecasts[valid] - y_test[valid])))
    else:
        rmse = mae = np.nan

    return {
        'ticker':    ticker,
        'forecasts': forecasts,
        'metrics':   {'Spółka': ticker, 'RMSE': rmse,
                      'MAE': mae, 'N_valid': int(valid.sum())},
    }


# ── Batch ─────────────────────────────────────────────────────────────────────

def process_batch(args):
    (batch_idx, tickers_batch, data_dict,
     train_size, retrain_every,
     checkpoint_dir, checkpoint_every) = args

    os.makedirs(checkpoint_dir, exist_ok=True)
    results = []

    for i, ticker in enumerate(tickers_batch):
        res = process_ticker(ticker, data_dict[ticker],
                             train_size, retrain_every)
        results.append(res)

        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(tickers_batch):
            cp = os.path.join(checkpoint_dir,
                              f'batch_{batch_idx:03d}_step_{i:04d}.pkl')
            pd.to_pickle(results, cp)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        'Model':        'LSTM',
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
    print(f"\n── LSTM Podsumowanie ──")
    print(f"RMSE mediana: {metrics['RMSE'].median():.8f}")
    print(f"MAE  mediana: {metrics['MAE'].median():.8f}")
    print(f"Spółek OK:    {metrics['RMSE'].notna().sum()}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()

    df = load_data()
    tickers   = df.columns.tolist()
    TEST_SIZE = len(df) - TRAIN_SIZE

    print(f"Spółek: {len(tickers)} | Train: {TRAIN_SIZE} | Test: {TEST_SIZE}")
    print(f"SEQ_LEN: {SEQ_LEN} | Retrenowanie co {RETRAIN_EVERY} dni "
          f"| LSTM units: {LSTM_UNITS} | Workery: {N_WORKERS}")

    done, existing = load_completed(CHECKPOINT_DIR)
    remaining = [t for t in tickers if t not in done]

    if done:
        print(f"Checkpointy: {len(done)} gotowych, {len(remaining)} pozostało")

    if not remaining:
        print("Wszystkie spółki gotowe — zapis wyników...")
        save_outputs(existing)
        exit()

    data_dict  = {t: df[t].values for t in remaining}
    batches    = [list(b) for b in np.array_split(remaining, N_WORKERS) if len(b)]
    batch_args = [
        (i, batch, data_dict, TRAIN_SIZE, RETRAIN_EVERY,
         CHECKPOINT_DIR, CHECKPOINT_EVERY)
        for i, batch in enumerate(batches)
    ]

    print(f"\nStart LSTM rolling... (~8-12h na CPU)\n")
    with mp.Pool(processes=N_WORKERS) as pool:
        results_list = pool.map(process_batch, batch_args)

    all_results = existing + [r for batch in results_list for r in batch]
    save_outputs(all_results)

    print(f"\nCzas całkowity: {(time.time()-t0)/3600:.2f} h")
    print("GOTOWE.")
