"""
log_returns.py - Obliczanie logarytmicznych stóp zwrotu

Wczytuje ceny zamknięcia i liczy dzienne logarytmiczne stopy zwrotu:
r_t = ln(P_t / P_{t-1}). Pierwszy wiersz (bez zdefiniowanej stopy
zwrotu) jest usuwany.

Wejście: dane1000close.xlsx
Wyjście: dane1000stopy.xlsx

----------------------------------------------------------------------

log_returns.py - Log return computation

Loads closing prices and computes daily log returns:
r_t = ln(P_t / P_{t-1}). The first row (with no defined return) is
dropped.

Input: dane1000close.xlsx
Output: dane1000stopy.xlsx
"""

import pandas as pd
import numpy as np

def calculate_log_returns(input_file, output_file):
    print(f"Wczytywanie danych z pliku: {input_file}...")
    
    # 1. Wczytanie pliku 
    try:
        df_close = pd.read_excel(input_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{input_file}' w obecnym folderze.")
        return

    # 2. Obliczenie logarytmicznych stóp zwrotu
    print("Obliczanie logarytmicznych stóp zwrotu...")
    df_stopy = np.log(df_close / df_close.shift(1))

    # 3. Usunięcie pierwszego wiersza 
    df_stopy = df_stopy.dropna(how='all')

    df_stopy.index = pd.to_datetime(df_stopy.index).strftime('%Y-%m-%d')
    # 4. Zapisanie do nowego pliku
    print(f"Zapisywanie wyników do pliku: {output_file}...")
    df_stopy.to_excel(output_file)
    
    print("Gotowe! Transformacja zakończona sukcesem.")

# --- Uruchomienie skryptu ---
if __name__ == "__main__":
    PLIK_WEJSCIOWY = 'dane1000close.xlsx' 
    PLIK_WYJSCIOWY = 'dane1000stopy.xlsx'
    
    calculate_log_returns(PLIK_WEJSCIOWY, PLIK_WYJSCIOWY)
