"""
test_arch.py - Test efektu ARCH na resztach modelu ARMA(1,1)

Dla każdej spółki szacuje model ARMA(1,1) na stopach zwrotu, a
następnie na resztach modelu przeprowadza test ARCH (het_arch, 10
opóźnień), aby sprawdzić obecność efektów ARCH (heteroskedastyczności
warunkowej) - H0: brak efektów ARCH.

Wejście: dane1000stopy.xlsx
Wyjście: wyniki_ARIMA_ARCH_1000_stopyzw.xlsx

----------------------------------------------------------------------

test_arch.py - ARCH-effect test on ARMA(1,1) residuals

For each company, fits an ARMA(1,1) model to the return series, then
runs the ARCH test (het_arch, 10 lags) on the model residuals to check
for ARCH effects (conditional heteroskedasticity) - H0: no ARCH
effects present.

Input: dane1000stopy.xlsx
Output: wyniki_ARIMA_ARCH_1000_stopyzw.xlsx
"""

import pandas as pd
import numpy as np
import time
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch 

# --- Ustawienia Plików ---
input_file = 'dane1000stopy.xlsx' 
output_file_arima_arch = 'wyniki_ARIMA_ARCH_1000_stopyzw.xlsx' 
ARIMA_ORDER = (1, 0, 1) # p=1, i=0, q=1

print(f"Wczytanie danych z pliku: {input_file}...")

try:
    df_returns = pd.read_excel(input_file)
    
    # Przygotowanie Danych
    col_data = None
    for col in df_returns.columns:
        if 'data' in col.lower() or 'date' in col.lower():
            col_data = col
            break
    
    if col_data:
        df_returns[col_data] = pd.to_datetime(df_returns[col_data])
        df_returns.set_index(col_data, inplace=True)
        
    for col in df_returns.columns:
        df_returns[col] = pd.to_numeric(df_returns[col].astype(str).str.replace(',', '.'), errors='coerce')

    df_returns.dropna(inplace=True)

    print(f"Liczba spółek do przeanalizowania: {len(df_returns.columns)}")
    print(f"Będzie szacowany model ARMA{ARIMA_ORDER[:-1]} i przeprowadzony test ARCH (het_arch)...")

    # --- Pętla Szacująca i Testująca ---
    wyniki_analizy = []
    start_time = time.time()

    for i, ticker in enumerate(df_returns.columns):
        series = df_returns[ticker].copy()
        
        if len(series) < 50: 
            wyniki_analizy.append({
                'Spółka': ticker, 'Status': 'BŁĄD: Zbyt mało danych', 'p_value_ARCH': None,
                'Wniosek_ARCH': None, 'Liczba obs. UŻYTYCH': len(series)
            })
            continue

        try:
            # 1. Szacowanie Modelu ARMA(1,1)
            model = ARIMA(series, order=ARIMA_ORDER)
            results = model.fit() 
            residuals = results.resid.dropna()
            
            ar_param = results.params.get('ar.L1')
            ma_param = results.params.get('ma.L1')

            # 2. Przeprowadzenie Testu ARCH
            lm_test = het_arch(residuals, nlags=10) 
            
            p_value_arch = lm_test[1] 
            
            # Wniosek z testu ARCH (H0: Brak efektów ARCH)
            if p_value_arch < 0.05:
                wniosek_arch = "ODRZUĆ H0 (Efekty ARCH obecne)"
            else:
                wniosek_arch = "NIE ODRZUCAJ H0 (Brak efektów ARCH)"
            
            # Zapis wyników
            wyniki_analizy.append({
                'Spółka': ticker,
                'Status': 'OK',
                'AR(1)': round(ar_param, 4) if ar_param is not None else None,
                'MA(1)': round(ma_param, 4) if ma_param is not None else None,
                'p_value_ARCH': round(p_value_arch, 6),
                'Wniosek_ARCH': wniosek_arch,
                'Liczba obs. UŻYTYCH': len(series)
            })
            
        except Exception as e:
            wyniki_analizy.append({
                'Spółka': ticker,
                'Status': f'BŁĄD: ARIMA/ARCH ({e})',
                'p_value_ARCH': None,
                'Wniosek_ARCH': None,
                'Liczba obs. UŻYTYCH': len(series)
            })

        if (i + 1) % 100 == 0:
            print(f"Przetworzono {i + 1} z {len(df_returns.columns)} spółek...")

    # --- Zapisanie Wyników do Excela ---
    df_wyniki = pd.DataFrame(wyniki_analizy)
    df_wyniki.to_excel(output_file_arima_arch, index=False)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("-" * 60)
    print(f"Zakończono analizę ARIMA i Test ARCH! Czas: {duration:.2f} s.")
    print(f"Raport zapisano w pliku: {output_file_arima_arch}")
    
    # Wyświetlenie statystyk
    arch_count = df_wyniki[df_wyniki['Wniosek_ARCH'] == 'ODRZUĆ H0 (Efekty ARCH obecne)']['Spółka'].count()
    total_ok = df_wyniki[df_wyniki['Status'] == 'OK']['Spółka'].count()
    
    if total_ok > 0:
        print(f"Udane szacowania: {total_ok}")
        print(f"Spółki z obecnymi efektami ARCH (p < 0.05): {arch_count} ({arch_count/total_ok * 100:.2f}%)")
    
    print("-" * 60)

except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{input_file}'.")
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd podczas ładowania lub przetwarzania danych: {e}")
