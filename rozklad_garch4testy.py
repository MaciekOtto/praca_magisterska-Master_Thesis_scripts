"""
rozklad_garch4testy.py - Diagnostyka rozkładu parametrów GARCH

Ponownie szacuje model GARCH(1,1) dla każdej spółki (na przeskalowanych
x100 stopach zwrotu), a następnie dla każdego parametru (omega, alpha,
beta, suma alpha+beta) przeprowadza test normalności i
dopasowuje najlepiej pasujący rozkład teoretyczny (spośród: normalny,
log-normalny, t-Studenta) metodą najmniejszych kwadratów (SSE) na
histogramie.

Wejście: dane1000stopy.xlsx
Wyjście: wyniki_GARCH_testy_statystyczne.xlsx (2 arkusze: parametry i
         wyniki testów statystycznych), analiza_rozkladow_garch1/*.png

----------------------------------------------------------------------

rozklad_garch4testy.py - GARCH parameter distribution diagnostics

Re-estimates a GARCH(1,1) model for each company (on returns scaled by
100), then for each parameter (omega, alpha, beta, alpha+beta sum)
runs a normality test and fits the best-matching
theoretical distribution (among normal, log-normal, Student's t) to
the parameter histogram via least-squares (SSE).

Input: dane1000stopy.xlsx
Output: wyniki_GARCH_testy_statystyczne.xlsx (2 sheets: parameters and
        statistical test results), analiza_rozkladow_garch1/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import scipy.stats as stats
import time
import os
import warnings

warnings.filterwarnings('ignore')

# --- Ustawienia Plików ---
input_file = 'dane1000stopy.xlsx'
output_excel = 'wyniki_GARCH_testy_statystyczne.xlsx'
output_folder = 'analiza_rozkladow_garch1'
os.makedirs(output_folder, exist_ok=True)

print(f"Wczytanie danych: {input_file}...")

try:
    df_returns = pd.read_excel(input_file)
    col_data = next((c for c in df_returns.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df_returns[col_data] = pd.to_datetime(df_returns[col_data])
        df_returns.set_index(col_data, inplace=True)
    
    df_returns = df_returns.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df_returns.dropna(how='all', inplace=True)

    wyniki_garch = []
    start_time = time.time()

    print(f"Szacowanie modeli GARCH(1,1) dla {len(df_returns.columns)} spółek...")

    for i, ticker in enumerate(df_returns.columns):
        series = df_returns[ticker].dropna()
        if len(series) < 50: continue

        try:
            # Standardowy model GARCH(1,1)
            am = arch_model(series * 100, mean='Constant', vol='Garch', p=1, q=1)
            res = am.fit(disp='off')

            wyniki_garch.append({
                'Spółka': ticker,
                'Omega (GARCH)': res.params['omega'],
                'Alpha (GARCH)': res.params['alpha[1]'],
                'Beta (GARCH)': res.params['beta[1]'],
                'Suma_AB': res.params['alpha[1]'] + res.params['beta[1]'],
                'Status': 'OK'
            })
        except:
            wyniki_garch.append({'Spółka': ticker, 'Status': 'BŁĄD'})

        if (i + 1) % 200 == 0:
            print(f"Przetworzono {i + 1} spółek...")

    df_results = pd.DataFrame(wyniki_garch)
    df_ok = df_results[df_results['Status'] == 'OK'].copy()

    # --- Testy ---
    print("\nAnaliza rozkładów parametrów...")
    
    parametry = ['Omega (GARCH)', 'Alpha (GARCH)', 'Beta (GARCH)', 'Suma_AB']
    dists_to_test = ['norm', 'lognorm', 't']
    raport_statystyczny = []

    for param_name in parametry:
        data = df_ok[param_name].dropna()
  
        q_low, q_high = data.quantile([0.01, 0.99])
        data_clean = data[(data > q_low) & (data < q_high)]

        # 1. Test Normalności
        stat, p_val = stats.normaltest(data_clean)
        
        # 2. Dopasowanie rozkładów i szukanie najlepszego (metoda SSE)
        y, x = np.histogram(data_clean, bins=50, density=True)
        x_mid = (x + np.roll(x, -1))[:-1] / 2.0
        
        best_dist = ""
        min_sse = np.inf

        for dist_name in dists_to_test:
            dist = getattr(stats, dist_name)
            params = dist.fit(data_clean)
            pdf = dist.pdf(x_mid, *params)
            sse = np.sum((y - pdf)**2)
            
            if sse < min_sse:
                min_sse = sse
                best_dist = dist_name

        raport_statystyczny.append({
            'Parametr': param_name,
            'Średnia': data_clean.mean(),
            'Skośność': data_clean.skew(),
            'p-value (Normalność)': round(p_val, 5),
            'Rozkład Normalny?': 'TAK' if p_val > 0.05 else 'NIE',
            'Najlepiej dopasowany': best_dist
        })

        # --- Wykres dopasowania ---
        plt.figure(figsize=(10, 6))
        sns.histplot(data_clean, kde=False, bins=40,stat="density", color='lightblue', label='Dane (GARCH)')
        
        # Rysowanie linii najlepszego rozkładu
        dist_best = getattr(stats, best_dist)
        params_best = dist_best.fit(data_clean)
        x_plot = np.linspace(data_clean.min(), data_clean.max(), 100)
        plt.plot(x_plot, dist_best.pdf(x_plot, *params_best), 'r-', lw=2, label=f'Dopasowanie: {best_dist}')
        
        plt.title(f'Test rozkładu dla parametru: {param_name}')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'Rozklad_{param_name}.png'))
        plt.close()

    # Zapis do Excela 
    df_stat = pd.DataFrame(raport_statystyczny)
    with pd.ExcelWriter(output_excel) as writer:
        df_ok.to_excel(writer, sheet_name='Parametry_GARCH', index=False)
        df_stat.to_excel(writer, sheet_name='Testy_Statystyczne', index=False)

    print(f"\nZAKOŃCZONO. Raport: {output_excel}")
    print(df_stat[['Parametr', 'Rozkład Normalny?', 'Najlepiej dopasowany']])

except Exception as e:
    print(f"Błąd: {e}")
