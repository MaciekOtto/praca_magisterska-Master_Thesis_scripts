"""
garch.py - Estymacja modelu GARCH(1,1) (pełna próba, firm-by-firm)

Dla każdej spółki szacuje model GARCH(1,1) (średnia stała, rozkład
normalny) na całej próbie stóp zwrotu. Zapisuje oszacowane parametry
(omega, alpha, beta) oraz warunek stacjonarności (alpha+beta < 1), a
także histogramy rozkładu tych parametrów.

Wejście: dane1000stopy.xlsx
Wyjście: wyniki_GARCH_1000.xlsx,
         wykresy_garch_parametry/*.png

----------------------------------------------------------------------

garch.py - GARCH(1,1) estimation (full sample, firm-by-firm)

Fits a GARCH(1,1) model (constant mean, normal distribution) to the
full return sample for each company. Saves the estimated parameters
(omega, alpha, beta) and the stationarity condition (alpha+beta < 1),
plus histograms of the parameter distributions.

Input: dane1000stopy.xlsx
Output: wyniki_GARCH_1000.xlsx,
        wykresy_garch_parametry/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import time

# --- Ustawienia Plików ---
input_file = 'dane1000stopy.xlsx'
output_file_garch = 'wyniki_GARCH_1000.xlsx' 
output_folder = 'wykresy_garch_parametry'

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

    print("\nszacowanie modeli GARCH(1,1) na stopach zwrotu...")

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
        sns.histplot(df_plot['Alpha (ARCH, B)'], kde=True, color='skyblue')
        plt.title('Rozkład parametru Alpha (ARCH) - Źródło klastrowania zmienności')
        plt.xlabel('Alpha ($\\alpha$)')
        plt.ylabel('Liczba Spółek')
        plt.savefig(os.path.join(output_folder, '01_Rozklad_Alpha_ARCH.png'))
        plt.close() 

        # --- Wykres 2: Beta (GARCH) ---
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot['Beta (GARCH, C)'], kde=True, color='lightcoral')
        plt.title('Rozkład parametru Beta (GARCH) - Trwałość zmienności')
        plt.xlabel('Beta ($\\beta$)')
        plt.ylabel('Liczba Spółek')
        plt.savefig(os.path.join(output_folder, '02_Rozklad_Beta_GARCH.png'))
        plt.close()

        # --- Wykres 3: Suma stabilności (Alpha + Beta) ---
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot['Suma A+B'], kde=True, color='lightgreen')
        plt.axvline(x=1.0, color='red', linestyle='--', label='Warunek stabilności (1.0)')
        plt.title('Rozkład Sumy Stabilności ($\\alpha + \\beta$)')
        plt.xlabel('Suma Alpha + Beta')
        plt.ylabel('Liczba Spółek')
        plt.legend()
        plt.savefig(os.path.join(output_folder, '03_Rozklad_Suma_Stabilnosci.png'))
        plt.close()
        
        print("Trzy wykresy dystrybucji parametrów zostały zapisane jako pliki PNG.")
        
    else:
        print("Brak udanych oszacowań do wygenerowania wykresów.")

except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{input_file}'.")
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd podczas ładowania lub przetwarzania danych: {e}")
