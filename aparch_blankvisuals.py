"""
aparch_blankvisuals.py - Alternatywny wariant aparch.py

Wariant aparch.py z tą samą metodą estymacji APARCH(1,1), różniący się
szczegółami wizualizacji i/lub nazwami plików wyjściowych. Pozostawiony
jako punkt odniesienia.

Wejście: dane1000stopy.xlsx
Wyjście: pliki wynikowe analogiczne do aparch.py (inne nazwy)

----------------------------------------------------------------------

aparch_blankvisuals.py - Alternate variant of aparch.py

A variant of aparch.py using the same APARCH(1,1) estimation method,
differing in visualization details and/or output filenames. Kept as a
reference.

Input: dane1000stopy.xlsx
Output: result files analogous to aparch.py (different names)
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
output_file_aparch = 'wyniki_APARCH_czysty.xlsx' 
output_folder = 'wykresy_aparch_parametry_czysty'
os.makedirs(output_folder, exist_ok=True)

print(f"Wczytuję dane z pliku: {input_file}...")

try:
    df_returns = pd.read_excel(input_file)
    
    # --- Przygotowanie Danych ---
    col_data = next((c for c in df_returns.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    
    if col_data:
        df_returns[col_data] = pd.to_datetime(df_returns[col_data])
        df_returns.set_index(col_data, inplace=True)
        
    # Konwersja na liczby i czyszczenie
    df_returns = df_returns.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df_returns.dropna(how='all', inplace=True)

    print(f"Liczba spółek do przeanalizowania: {len(df_returns.columns)}")

    wyniki_aparch = []
    start_time = time.time()

    print("\nszacowanie modeli APARCH(1,1)...")

    for i, ticker in enumerate(df_returns.columns):
        series = df_returns[ticker].dropna()
        
        series_scaled = series * 100
        
        if len(series_scaled) < 100: 
            continue

        try:
            am = arch_model(series_scaled, mean='Constant', vol='APARCH', p=1, o=1, q=1, rescale=True)
            res = am.fit(disp='off', show_warning=False)

            p = res.params
            
            omega = p.get('omega', np.nan)
            alpha = p.get('alpha[1]', np.nan)
            gamma = p.get('gamma[1]', np.nan)
            beta = p.get('beta[1]', np.nan)
            delta = p.get('delta', np.nan) 
            
            wyniki_aparch.append({
                'Spółka': ticker,
                'Omega (APARCH)': round(omega, 6) if not np.isnan(omega) else None,
                'Alpha (APARCH)': round(alpha, 6) if not np.isnan(alpha) else None,
                'Gamma (APARCH)': round(gamma, 6) if not np.isnan(gamma) else None,
                'Beta (APARCH)': round(beta, 6) if not np.isnan(beta) else None,
                'Delta (APARCH)': round(delta, 6) if not np.isnan(delta) else None,
                'AIC': res.aic,
                'Status': 'OK'
            })
            
        except Exception as e:
            wyniki_aparch.append({
                'Spółka': ticker,
                'Status': f'BŁĄD: {str(e)[:50]}'
            })

        if (i + 1) % 100 == 0:
            print(f"Przetworzono {i + 1} spółek...")

    # 2. Zapisanie Wyników
    df_wyniki = pd.DataFrame(wyniki_aparch)
    df_wyniki.to_excel(output_file_aparch, index=False)
    
    print(f"\nZakończono! Wyniki zapisano w: {output_file_aparch}")

    # 3. Tworzenie Wykresów
    df_plot = df_wyniki[df_wyniki['Status'] == 'OK'].copy()
    
    if not df_plot.empty:
        # Lista parametrów do narysowania
        params = [
            ('Omega (APARCH)','gray','Bazowy poziom wariancji.png'),
            ('Alpha (APARCH)', 'skyblue', '01_Rozklad_Alpha.png'),
            ('Beta (APARCH)', 'salmon', '02_Rozklad_Beta.png'),
            ('Gamma (APARCH)', 'purple', '03_Rozklad_Gamma.png'),
            ('Delta (APARCH)', 'orange', '04_Rozklad_Delta.png')
        ]

        for col, color, filename in params:
            plt.figure(figsize=(10, 6))
            data = df_plot[col].dropna()
            q_low, q_high = data.quantile([0.01, 0.99])
            data_filtered = data[(data > q_low) & (data < q_high)]
            
            sns.histplot(data_filtered, kde=False, bins=40, color=color)
            
            if 'Gamma' in col:
                plt.title('Rozkład parametru Gamma (APARCH)')
            elif 'Delta' in col:
                plt.title('Rozkład parametru Delta (APARCH)')
            else:
                plt.title(f'Rozkład parametru {col}')
            
            plt.legend()
            plt.savefig(os.path.join(output_folder, filename))
            plt.close()
        
        print(f"Wykresy (Alpha, Beta, Gamma, Delta) zapisano w folderze: {output_folder}")

except Exception as e:
    print(f"Błąd krytyczny: {e}")
