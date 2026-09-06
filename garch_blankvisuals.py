"""
garch_blankvisuals.py - Alternatywny wariant garch.py

Niemal identyczny z garch.py (ta sama estymacja GARCH(1,1) na pełnej
próbie), z drobnymi różnicami w zapisywanych wykresach (m.in. dodatkowy
histogram parametru Omega, inne ustawienia KDE) i innymi nazwami
plików wyjściowych. Pozostawiony jako punkt odniesienia.

Wejście: dane1000stopy.xlsx
Wyjście: wyniki_GARCH_czyste.xlsx, wykresy_garch_czyste/*.png

----------------------------------------------------------------------

garch_blankvisuals.py - Alternate variant of garch.py

Nearly identical to garch.py (same full-sample GARCH(1,1) estimation),
with minor differences in the plots produced (e.g. an extra Omega
histogram, different KDE settings) and different output filenames.
Kept as a reference.

Input: dane1000stopy.xlsx
Output: wyniki_GARCH_czyste.xlsx, wykresy_garch_czyste/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import time

# --- Ustawienia Plików ---
input_file = 'dane1000stopy.xlsx'
output_file_garch = 'wyniki_GARCH_czyste.xlsx' 
output_folder = 'wykresy_garch_czyste'

print(f"Wczytanie danych z pliku: {input_file}...")

try:
    df_returns = pd.read_excel(input_file)

    col_data = None
    for col in df_returns.columns:
        if 'data' in col.lower() or 'date' in col.lower():
            col_data = col
            break
    
    if col_data:
        df_returns[col_data] = pd.to_datetime(df_returns[col_data])
        df_returns.set_index(col_data, inplace=True)
        
    df_returns.dropna(inplace=True) 

    for col in df_returns.columns:
        df_returns[col] = pd.to_numeric(df_returns[col].astype(str).str.replace(',', '.'), errors='coerce')

    df_returns.dropna(inplace=True)

    print(f"Liczba spółek do przeanalizowania: {len(df_returns.columns)}")

    # 1. Oszacowanie Modelu GARCH(1,1)
    
    wyniki_garch = []
    start_time = time.time()

    print("\nszacowanie modeli GARCH(1,1) na gotowych stopach zwrotu...")

    for i, ticker in enumerate(df_returns.columns):
        series = df_returns[ticker].copy()
        
        if len(series) < 50 or series.isnull().any(): 
            wyniki_garch.append({
                'Spółka': ticker,
                'Omega (A)': None, 'Alpha (ARCH, B)': None, 'Beta (GARCH, C)': None,
                'Suma A+B': None, 'Liczba obs. UŻYTYCH': len(series),
                'Czy stacjonarny (A+B<1)': 'N/D',
                'Status': 'BŁĄD: Zbyt mało lub braki danych po czyszczeniu'
            })
            continue

        try:
            am = arch_model(series, mean='Constant', vol='Garch', p=1, q=1)
            res = am.fit(disp='off')

            omega = res.params['omega']
            alpha = res.params['alpha[1]']
            beta = res.params['beta[1]']
            suma_stabilnosci = alpha + beta
            
            wyniki_garch.append({
                'Spółka': ticker,
                'Omega (A)': round(omega, 8),
                'Alpha (ARCH, B)': round(alpha, 6),
                'Beta (GARCH, C)': round(beta, 6),
                'Suma A+B': round(suma_stabilnosci, 6),
                'Liczba obs. UŻYTYCH': len(series),
                'Czy stacjonarny (A+B<1)': 'TAK' if suma_stabilnosci < 1.0 else 'NIE',
                'Status': 'OK'
            })
            
        except Exception as e:
            wyniki_garch.append({
                'Spółka': ticker,
                'Omega (A)': None, 'Alpha (ARCH, B)': None, 'Beta (GARCH, C)': None,
                'Suma A+B': None, 'Liczba obs. UŻYTYCH': len(series),
                'Czy stacjonarny (A+B<1)': 'N/D',
                'Status': f'BŁĄD: Konwergencja ({e})'
            })

        if (i + 1) % 100 == 0:
            print(f"Przetworzono {i + 1} z {len(df_returns.columns)} spółek...")

    # 2. Zapisanie Wyników do Excela
    df_wyniki_garch = pd.DataFrame(wyniki_garch)
    df_wyniki_garch.to_excel(output_file_garch, index=False)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("-" * 50)
    print(f"Zakończono szacowanie GARCH(1,1)! Czas: {duration:.2f} s.")
    print(f"Raport zapisano w pliku: {output_file_garch}")
    print("-" * 50)
 
    import os
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"Wykresy zostaną zapisane w folderze: {output_folder}")

    df_plot = df_wyniki_garch[df_wyniki_garch['Status'] == 'OK'].copy()
    
    if not df_plot.empty:
        
        # --- Wykres 1: Alpha (ARCH) ---
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot['Alpha (ARCH, B)'], kde=False, bins=40,color='skyblue')
        plt.title('Rozkład parametru Alpha (GARCH)')
        plt.xlabel('Alpha (GARCH)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_folder, '01_Rozklad_Alpha_ARCH.png'))
        plt.close() # Zamknięcie figury, aby zwolnić pamięć

        # --- Wykres 2: Beta (GARCH) ---
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot['Beta (GARCH, C)'], kde=False, bins=40,color='salmon')
        plt.title('Rozkład parametru Beta (GARCH)')
        plt.xlabel('Beta (GARCH)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_folder, '02_Rozklad_Beta_GARCH.png'))
        plt.close()

        # --- Wykres 3: Suma stabilności (Alpha + Beta) ---
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot['Suma A+B'], kde=True, bins=40,color='lightgreen')
        plt.axvline(x=1.0, color='red', linestyle='--', label='Warunek stabilności (1.0)')
        plt.title('Rozkład Sumy Stabilności ($\\alpha + \\beta$)')
        plt.xlabel('Suma Alpha + Beta')
        plt.ylabel('Liczba Spółek')
        plt.legend()
        plt.savefig(os.path.join(output_folder, '03_Rozklad_Suma_Stabilnosci.png'))
        plt.close()
        
         # --- Wykres 3: Omega (GARCH) ---
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot['Omega (A)'], kde=False,bins=40, color='gray')
        plt.title('Rozkład parametru Omega (GARCH)')
        plt.xlabel('Omega (GARCH)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_folder, '02_Rozklad_omega_GARCH.png'))
        plt.close()

        print("Trzy wykresy dystrybucji parametrów zostały zapisane jako pliki PNG.")
        
    else:
        print("Brak udanych oszacowań do wygenerowania wykresów.")

except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{input_file}'.")
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd podczas ładowania lub przetwarzania danych: {e}")
