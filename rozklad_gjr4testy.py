"""
rozklad_gjr4testy.py - Diagnostyka rozkładu parametrów GJR-GARCH

Analogicznie do rozklad_garch4testy.py, ale dla modelu GJR-GARCH(1,1)
(rozkład normalny, `o=1`): ponowna estymacja parametrów dla każdej
spółki i testy dopasowania rozkładu dla każdego parametru modelu.

Wejście: dane1000stopy.xlsx
Wyjście: wyniki_GJR_z_testami_rozkładów.xlsx,
         analiza_statystyczna_parametrów/*.png

----------------------------------------------------------------------

rozklad_gjr4testy.py - GJR-GARCH parameter distribution diagnostics

Analogous to rozklad_garch4testy.py, but for the GJR-GARCH(1,1) model
(normal distribution, `o=1`): re-estimates the parameters for each
company and runs distribution-fit tests for each model parameter.

Input: dane1000stopy.xlsx
Output: wyniki_GJR_z_testami_rozkładów.xlsx,
        analiza_statystyczna_parametrów/*.png
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

# --- Ustawienia ---
input_file = 'dane1000stopy.xlsx'
output_excel = 'wyniki_GJR_z_testami_rozkładów.xlsx'
output_folder = 'analiza_statystyczna_parametrów'
os.makedirs(output_folder, exist_ok=True)

try:
    print(f"Wczytanie danych: {input_file}...")
    df = pd.read_excel(input_file)
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df[col_data] = pd.to_datetime(df[col_data])
        df.set_index(col_data, inplace=True)
    
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce')).dropna(how='all')

    wyniki = []
    print(f"analiza dla {len(df.columns)} spółek...")

    for i, ticker in enumerate(df.columns):
        series = df[ticker].dropna()
        if len(series) < 150: continue

        try:
            # Model GJR-GARCH 
            am = arch_model(series * 100, mean='Constant', vol='Garch', p=1, o=1, q=1, dist='Normal')
            res = am.fit(disp='off')
            
            p = res.params
            wyniki.append({
                'Spółka': ticker,
                'Omega': p['omega'],
                'Alpha': p['alpha[1]'],
                'Gamma': p['gamma[1]'],
                'Beta': p['beta[1]'],
                'Persystencja': p['alpha[1]'] + p['beta[1]'] + 0.5 * p['gamma[1]'],
                'Status': 'OK'
            })
        except:
            wyniki.append({'Spółka': ticker, 'Status': 'BŁĄD'})

        if (i + 1) % 200 == 0:
            print(f"Przetworzono {i + 1} spółek...")

    df_res = pd.DataFrame(wyniki)
    df_ok = df_res[df_res['Status'] == 'OK'].copy()

    # --- Testy ---
    print("\nRozpoczynam testowanie rozkładów parametrów...")
    
    parametry = ['Omega', 'Alpha', 'Gamma', 'Beta', 'Persystencja']
    dist_names = ['norm', 'lognorm', 't'] # Normalny, Log-normalny, t-Studenta
    
    raport_stat = []

    for param in parametry:
        data = df_ok[param].dropna()
        q_low, q_high = data.quantile([0.025, 0.975])
        data_clean = data[(data > q_low) & (data < q_high)]

        # 1. Test Normalności 
        stat, p_val = stats.normaltest(data_clean)
        
        # 2. Szukanie najlepszego rozkładu (Metoda najmniejszych kwadratów dla histogramu)
        y, x = np.histogram(data_clean, bins=50, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        best_dist = ""
        best_sse = np.inf

        for dist_name in dist_names:
            dist = getattr(stats, dist_name)
            params = dist.fit(data_clean)
            
            pdf = dist.pdf(x, *params)
            sse = np.sum((y - pdf)**2)
            
            if sse < best_sse:
                best_sse = sse
                best_dist = dist_name

        raport_stat.append({
            'Parametr': param,
            'Średnia': data_clean.mean(),
            'Skośność': data_clean.skew(),
            'Kurtoza': data_clean.kurtosis(),
            'p-value (Normalność)': p_val,
            'Jest Normalny?': 'TAK' if p_val > 0.05 else 'NIE',
            'Najlepszy rozkład': best_dist
        })

        # --- Wykres dopasaowania ---
        plt.figure(figsize=(10, 6))
        sns.histplot(data_clean, kde=False, stat="density", color='lightgray', label='Dane empiryczne')
        
        # Rysowanie najlepszego rozkładu
        dist = getattr(stats, best_dist)
        params = dist.fit(data_clean)
        x_plot = np.linspace(data_clean.min(), data_clean.max(), 100)
        plt.plot(x_plot, dist.pdf(x_plot, *params), 'r-', lw=2, label=f'Dopasowany: {best_dist}')
        
        plt.title(f'Rozkład parametru {param} (Najlepsze dopasowanie: {best_dist})')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'Dopasowanie_{param}.png'))
        plt.close()

    # Zapis wyników
    df_stat = pd.DataFrame(raport_stat)
    with pd.ExcelWriter(output_excel) as writer:
        df_ok.to_excel(writer, sheet_name='Parametry_Spółek', index=False)
        df_stat.to_excel(writer, sheet_name='Testy_Statystyczne', index=False)

    print(f"\nANALIZA ZAKOŃCZONA.")
    print(df_stat[['Parametr', 'Jest Normalny?', 'Najlepszy rozkład']])

except Exception as e:
    print(f"Błąd: {e}")
