"""
test_adf.py - Test stacjonarności ADF (Augmented Dickey-Fuller)

Dla każdej spółki przeprowadza rozszerzony test Dickeya-Fullera na
szeregu stóp zwrotu, aby sprawdzić stacjonarność (H0: szereg
niestacjonarny). Wynik uznaje się za stacjonarny przy p-value < 0.05.

Wejście: dane1000stopy.xlsx
Wyjście: wyniki_ADF_1000_stp.xlsx (statystyka ADF, p-value, wniosek)

----------------------------------------------------------------------

test_adf.py - ADF (Augmented Dickey-Fuller) stationarity test

Runs the Augmented Dickey-Fuller test on each company's return series
to test for stationarity (H0: the series is non-stationary). A series
is considered stationary when p-value < 0.05.

Input: dane1000stopy.xlsx
Output: wyniki_ADF_1000_stp.xlsx (ADF statistic, p-value, conclusion)
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import time

# Nazwa pliku wejściowego
input_file = 'dane1000stopy.xlsx'
# Nazwa pliku wyjściowego z wynikami
output_file = 'wyniki_ADF_1000_stp.xlsx'

print(f"Wczytanie danych z pliku: {input_file}...")

try:
    df = pd.read_excel(input_file)

    col_data = None
    for col in df.columns:
        if 'data' in col.lower() or 'date' in col.lower():
            col_data = col
            break
    
    if col_data:
        df[col_data] = pd.to_datetime(df[col_data])
        df.set_index(col_data, inplace=True)
        print(f"Ustawiono kolumnę '{col_data}' jako indeks czasowy.")
    else:
        print("UWAGA: Nie znaleziono kolumny z datą.")

    print(f"Liczba spółek do przeanalizowania: {len(df.columns)}")
    print("Rozpoczynanie testu ADF...")

    wyniki_testow = []
    start_time = time.time()

    for i, ticker in enumerate(df.columns):
        try:
            series = df[ticker].dropna()
            if series.dtype == 'object':
                series = series.astype(str).str.replace(',', '.').astype(float)

            # Test ADF
            result = adfuller(series)
            adf_stat = result[0]
            p_value = result[1]
            usedlag = result[2]
            nobs = result[3]
            
            # Zapis wyniku
            wyniki_testow.append({
                'Spółka': ticker,
                'Statystyka ADF': round(adf_stat, 4),
                'p-value': round(p_value, 6),
                'Czy stacjonarny (p<0.05)': 'TAK' if p_value < 0.05 else 'NIE',
                'Liczba obs.': nobs
            })
        except Exception as e:
            print(f"Błąd dla spółki {ticker}: {e}")
            wyniki_testow.append({
                'Spółka': ticker,
                'Statystyka ADF': None,
                'p-value': None,
                'Czy stacjonarny (p<0.05)': 'BŁĄD',
                'Liczba obs.': 0
            })

        if (i + 1) % 100 == 0:
            print(f"Przetworzono {i + 1} z {len(df.columns)} spółek...")

    # Zapis do Excela
    df_wyniki = pd.DataFrame(wyniki_testow)
    df_wyniki.to_excel(output_file, index=False)
    
    end_time = time.time()
    duration = end_time - start_time

    print("-" * 30)
    print(f"Zakończono! Przeanalizowano {len(df.columns)} szeregów w {duration:.2f} s.")
    print(f"Wyniki zapisano w pliku: {output_file}")

except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{input_file}'. Upewnij się, że jest w tym samym folderze co skrypt.")
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd: {e}")
