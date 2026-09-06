"""
gjrgarch.py - Estymacja modelu GJR-GARCH(1,1) (pełna próba, firm-by-firm)

Dla każdej spółki szacuje model GJR-GARCH(1,1) (GARCH z dodatkowym
parametrem asymetrii `o=1`, wychwytującym efekt dźwigni) na stopach
zwrotu. Zapisuje oszacowane parametry oraz histogramy ich rozkładów.

Wejście: dane1000stopy.xlsx
Wyjście: wyniki_GJRGARCH_1000.xlsx, wykresy_gjrgarch_parametry/*.png

----------------------------------------------------------------------

gjrgarch.py - GJR-GARCH(1,1) estimation (full sample, firm-by-firm)

Fits a GJR-GARCH(1,1) model (GARCH with an additional asymmetry
parameter `o=1`, capturing the leverage effect) to each company's
return series. Saves the estimated parameters and histograms of their
distributions.

Input: dane1000stopy.xlsx
Output: wyniki_GJRGARCH_1000.xlsx, wykresy_gjrgarch_parametry/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import time
import os
import warnings

warnings.filterwarnings('ignore')

# --- Ustawienia Plików ---
input_file = 'dane1000stopy.xlsx'
output_file_gjr = 'wyniki_GJRGARCH_1000.xlsx' 
output_folder = 'wykresy_gjrgarch_parametry'
os.makedirs(output_folder, exist_ok=True)

print(f"Wczytanie dane z pliku: {input_file}...")

try:
    df_returns = pd.read_excel(input_file)
    
    # --- Przygotowanie Danych ---
    col_data = next((col for col in df_returns.columns if 'data' in col.lower() or 'date' in col.lower()), None)
    
    if col_data:
        df_returns[col_data] = pd.to_datetime(df_returns[col_data])
        df_returns.set_index(col_data, inplace=True)
        
    df_returns = df_returns.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df_returns.dropna(how='all', inplace=True)

    wyniki_gjr = []
    start_time = time.time()

    print(f"szacowanie dla {len(df_returns.columns)} spółek...")

    for i, ticker in enumerate(df_returns.columns):
        series = df_returns[ticker].dropna()
        
        if len(series) < 100: 
            wyniki_gjr.append({'Spółka': ticker, 'Status': 'BŁĄD: Za mało danych'})
            continue

        try:
            # Model GJR-GARCH: p=1, o=1, q=1
            am = arch_model(series, mean='Constant', vol='Garch', p=1, o=1, q=1, dist='Normal')
            
            res = am.fit(disp='off')

            p = res.params
            alpha = p.get('alpha[1]', 0)
            gamma = p.get('gamma[1]', 0)
            beta = p.get('beta[1]', 0)
            omega = p.get('omega', 0)
            
            persistence = alpha + beta + 0.5 * gamma
            
            wyniki_gjr.append({
                'Spółka': ticker,
                'Omega': round(omega, 8),
                'Alpha (ARCH)': round(alpha, 6),
                'Gamma (Asymetria)': round(gamma, 6),
                'Beta (GARCH)': round(beta, 6),
                'Persystencja': round(persistence, 6),
                'AIC': res.aic,
                'Status': 'OK'
            })
            
        except Exception as e:
            wyniki_gjr.append({
                'Spółka': ticker,
                'Status': f'BŁĄD: {str(e)[:40]}'
            })

        if (i + 1) % 100 == 0:
            print(f"Przetworzono {i + 1} spółek...")

    # Zapis do Excela
    df_wyniki = pd.DataFrame(wyniki_gjr)
    df_wyniki.to_excel(output_file_gjr, index=False)
    
    print(f"\nSzacowanie zakończone. Czas: {time.time() - start_time:.2f} s.")

    # --- Generowanie wykresów --- 
    df_plot = df_wyniki[df_wyniki['Status'] == 'OK'].copy()
    
    if not df_plot.empty:
        param_list = [
            ('Omega', 'gray', 'Rozkład Omegi'),
            ('Alpha (ARCH)', 'skyblue', 'Rozkład Alfy'),
            ('Gamma (Asymetria)', 'purple', 'Rozkład Gammy (Asymetria)'),
            ('Beta (GARCH)', 'salmon', 'Rozkład Bety')
        ]

        for col, color, title in param_list:
            plt.figure(figsize=(10, 6))
            data = df_plot[col].dropna()
            q_low, q_high = data.quantile([0.01, 0.99])
            data_filtered = data[(data >= q_low) & (data <= q_high)]
            
            sns.histplot(data_filtered, kde=True, color=color)
            plt.axvline(data_filtered.mean(), color='red', ls='--', label=f'Śr: {data_filtered.mean():.4f}')
            if 'Gamma' in col: plt.axvline(0, color='black', lw=1)
            
            plt.title(title)
            plt.legend()
            plt.savefig(os.path.join(output_folder, f'{col[:5]}.png'))
            plt.close()
        
        print(f"Wykresy zapisano w folderze: {output_folder}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")
