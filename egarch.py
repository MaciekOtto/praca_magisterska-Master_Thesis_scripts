"""
egarch.py - Estymacja modelu EGARCH(1,1) (pełna próba, firm-by-firm)

Dla każdej spółki szacuje model EGARCH(1,1) na przeskalowanych stopach
zwrotu (x100, dla stabilności numerycznej estymacji). Spółki z ponad
80% zer w danych lub mniej niż 100 obserwacjami są pomijane. Zapisuje
parametry modelu oraz histogramy ich rozkładów.

Wejście: dane1000stopy.xlsx
Wyjście: WYNIKI_EGARCH.xlsx, wykresy_egarch_parametr/*.png

----------------------------------------------------------------------

egarch.py - EGARCH(1,1) estimation (full sample, firm-by-firm)

Fits an EGARCH(1,1) model to each company's return series, scaled by
100 for numerical stability of the estimation. Companies with more
than 80% zeros in the data or fewer than 100 observations are skipped.
Saves the model parameters and histograms of their distributions.

Input: dane1000stopy.xlsx
Output: WYNIKI_EGARCH.xlsx, wykresy_egarch_parametr/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import os
import warnings

warnings.filterwarnings('ignore')

# --- Ustawienia ---
input_file = 'dane1000stopy.xlsx'
output_excel = 'WYNIKI_EGARCH.xlsx'
output_folder = 'wykresy_egarch_parametr'
os.makedirs(output_folder, exist_ok=True)

try:
    print(f"Wczytuję dane z: {input_file}...")
    df = pd.read_excel(input_file)
    
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df.set_index(pd.to_datetime(df[col_data]), inplace=True)
    
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df = df.dropna(how='all')

    wyniki = []
    print(f"Analiza {len(df.columns)} spółek modelem EGARCH...")

    for i, ticker in enumerate(df.columns):
        series = df[ticker].dropna()

        if (series == 0).sum() / len(series) > 0.8:
            wyniki.append({'Spółka': ticker, 'Status': 'BŁĄD: Zbyt dużo zer w danych'})
            continue

        if len(series) < 100:
            continue

        # SKALOWANIE
        series_scaled = series * 100

        try:
            
            am = arch_model(series_scaled, mean='Constant', vol='EGARCH', p=1, o=1, q=1, rescale=True)
            
            # Dopasowanie modelu
            res = am.fit(disp='off', show_warning=False)

            # Pobieranie parametrów
            p = res.params
            g_val = next((p[k] for k in p.index if 'gamma' in k.lower()), np.nan)
            a_val = next((p[k] for k in p.index if 'alpha' in k.lower()), np.nan)
            b_val = next((p[k] for k in p.index if 'beta' in k.lower()), np.nan)
            
            wyniki.append({
                'Spółka': ticker,
                'Omega': p.get('omega', np.nan),
                'Alpha': a_val,
                'Gamma': g_val,
                'Beta': b_val,
                'AIC': res.aic,
                'BIC': res.bic,
                'Status': 'OK'
            })
        except Exception as e:
            
            wyniki.append({'Spółka': ticker, 'Status': f'BŁĄD: {str(e)[:40]}'})

        if (i + 1) % 100 == 0:
            print(f"Przetworzono {i+1} spółek...")

    # Zapis do Excela
    df_res = pd.DataFrame(wyniki)
    df_res.to_excel(output_excel, index=False)
    print(f"\n Zapisano wyniki do: {output_excel}")

    # --- WYKRESY ---
    df_plot = df_res[df_res['Status'] == 'OK'].copy()
    if not df_plot.empty:
        print("Tworzenie wykresy parametrów...")
        for col, title, color in [('Omega','Poziom wariancji bazowej-LOG','gray'),
                                   ('Alpha', 'Wpływ szoku (Alpha)', 'skyblue'), 
                                   ('Gamma', 'Asymetria (Gamma)', 'purple'), 
                                   ('Beta', 'Trwałość (Beta)', 'salmon')]:
            plt.figure(figsize=(10,6))
            data_to_plot = df_plot[col].dropna()
            
            # Usuwanie outlierów
            q_low = data_to_plot.quantile(0.01)
            q_hi  = data_to_plot.quantile(0.99)
            data_filtered = data_to_plot[(data_to_plot > q_low) & (data_to_plot < q_hi)]
            
            sns.histplot(data_filtered, kde=True, color=color)
            plt.axvline(data_filtered.mean(), color='red', linestyle='--', label=f'Śr: {data_filtered.mean():.3f}')
            if col == 'Gamma': plt.axvline(0, color='black')
            plt.title(f"Rozkład parametru {title}")
            plt.legend()
            plt.savefig(os.path.join(output_folder, f"Rozklad_{col}.png"))
            plt.close()
        print(f" Wykresy gotowe w folderze: {output_folder}")
    else:
        print(" Uwaga: Brak udanych oszacowań OK. Sprawdź Excel.")

except Exception as e:
    print(f" Błąd krytyczny skryptu: {e}")
