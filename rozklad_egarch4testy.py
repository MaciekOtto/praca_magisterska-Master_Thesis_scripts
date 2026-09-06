"""
rozklad_egarch4testy.py - Diagnostyka rozkładu parametrów EGARCH

Analogicznie do rozklad_garch4testy.py, ale dla modelu EGARCH(1,1):
ponowna estymacja parametrów dla każdej spółki i testy dopasowania
rozkładu (normalność + najlepszy rozkład teoretyczny) dla każdego
parametru modelu.

Wejście: dane1000stopy.xlsx
Wyjście: wyniki_EGARCH4TEST.xlsx, analiza_statystyczna_egarch41/*.png

----------------------------------------------------------------------

rozklad_egarch4testy.py - EGARCH parameter distribution diagnostics

Analogous to rozklad_garch4testy.py, but for the EGARCH(1,1) model:
re-estimates the parameters for each company and runs distribution-fit
tests (normality + best theoretical distribution) for each model
parameter.

Input: dane1000stopy.xlsx
Output: wyniki_EGARCH4TEST.xlsx, analiza_statystyczna_egarch41/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import scipy.stats as stats
import os
import time
import warnings

warnings.filterwarnings('ignore')

# --- Ustawienia ---
input_file = 'dane1000stopy.xlsx'
output_excel = 'wyniki_EGARCH4TEST.xlsx'
output_folder = 'analiza_statystyczna_egarch41'
os.makedirs(output_folder, exist_ok=True)

try:
    print(f"Wczytanie danych: {input_file}...")
    df = pd.read_excel(input_file)
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df.set_index(pd.to_datetime(df[col_data]), inplace=True)
    
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce')).dropna(how='all')

    wyniki_raw = []
    print(f"Analiza {len(df.columns)} spółek modelem EGARCH(1,1)...")

    for i, ticker in enumerate(df.columns):
        series = df[ticker].dropna()
        if len(series) < 150: continue
        
        try:
            # EGARCH(1,1) 
            am = arch_model(series * 100, mean='Constant', vol='EGARCH', p=1, o=1, q=1, rescale=True)
            res = am.fit(disp='off')
            
            p = res.params
            p_vals=res.pvalues
            wyniki_raw.append({
                'Spółka': ticker,
                'Omega (EGARCH)': p.get('omega'),
                'Alpha (EGARCH)': p.get('alpha[1]'),
                'Gamma (EGARCH)': p.get('gamma[1]'),
                'Beta (EGARCH)': p.get('beta[1]'),
                'Czy_Gamma_Istotna': 'TAK' if p_vals['gamma[1]'] < 0.05 else 'NIE',
                'AIC': res.aic,
                'Status': 'OK'
            })
        except:
            wyniki_raw.append({'Spółka': ticker, 'Status': 'BŁĄD'})

        if (i + 1) % 200 == 0:
            print(f"Przetworzono {i+1} spółek...")

    df_res = pd.DataFrame(wyniki_raw)
    df_ok = df_res[df_res['Status'] == 'OK'].copy()

    # --- Testy ---
    print("\nPrzeprowadzam testy p-value dla rozkładów: Normalny, t-Studenta, Log-normalny, F...")
    
    
    parametry = ['Alpha (EGARCH)', 'Beta (EGARCH)', 'Omega (EGARCH)','Gamma (EGARCH)']
    raport_stat = []

    for param in parametry:
        data = df_ok[param].dropna()
        q_low, q_high = data.quantile([0.01, 0.99])
        data_clean = data[(data > q_low) & (data < q_high)]

        # Słownik na wyniki p-value
        p_values = {}

        # 1. Test Rozkładu Normalnego
        params_norm = stats.norm.fit(data_clean)
        _, p_values['p-val Normalny'] = stats.kstest(data_clean, 'norm', args=params_norm)

        # 2. Test Rozkładu t-Studenta
        params_t = stats.t.fit(data_clean)
        _, p_values['p-val t-Student'] = stats.kstest(data_clean, 't', args=params_t)

        # 3. Test Rozkładu Log-normalnego
        data_pos = data_clean + 0.00001 if data_clean.min() <= 0 else data_clean
        params_log = stats.lognorm.fit(data_pos)
        _, p_values['p-val Log-normalny'] = stats.kstest(data_pos, 'lognorm', args=params_log)

        # 4. Test Rozkładu F (Snedecora)
        params_f = stats.f.fit(data_pos)
        _, p_values['p-val F'] = stats.kstest(data_pos, 'f', args=params_f)

        # Dodanie wyników do raportu
        res_row = {
            'Parametr': param,
            'Średnia': data_clean.mean(),
            'Skośność': data_clean.skew(),
            'Kurtoza': data_clean.kurtosis()
        }
        res_row.update(p_values)
        raport_stat.append(res_row)

        # --- Wybór najlepszego rozkładu (Najwyższe p-value) ---
        mapping = {
            'p-val Normalny': ('norm', 'Normalny', 'blue'),
            'p-val t-Student': ('t', 't-Student', 'red'),
            'p-val Log-normalny': ('lognorm', 'Log-normalny', 'green'),
            'p-val F': ('f', 'F-Snedecora', 'orange')
        }
        
        # Znajdujemy klucz z najwyższym p-value
        best_key = max(p_values, key=p_values.get)
        best_dist_code, best_name, best_color = mapping[best_key]
        
        res_row['Rekomendowany Rozkład'] = best_name
        res_row['Najwyższe p-value'] = p_values[best_key]

        # --- Wykres z najlepszą krzywą ---
        plt.figure(figsize=(10, 6))
        # Histogram danych
        sns.histplot(data_clean, kde=False,bins=40, stat="density", color='lightgray', label='Dane empiryczne')
        
        # Dopasowanie i rysowanie krzywej
        dist_func = getattr(stats, best_dist_code)

        if best_dist_code in ['lognorm', 'f']:
            d_plot = data_clean + 0.00001 if data_clean.min() <= 0 else data_clean
            params = dist_func.fit(d_plot)
            x = np.linspace(d_plot.min(), d_plot.max(), 100)
            plt.plot(x, dist_func.pdf(x, *params), color=best_color, lw=2.5, 
                     label=f'Najlepsze dopasowanie: {best_name}\n(p-val: {p_values[best_key]:.4f})')
        else:
            params = dist_func.fit(data_clean)
            x = np.linspace(data_clean.min(), data_clean.max(), 100)
            plt.plot(x, dist_func.pdf(x, *params), color=best_color, lw=2.5, 
                     label=f'Najlepsze dopasowanie: {best_name}\n(p-val: {p_values[best_key]:.4f})')

        plt.title(f'EGARCH: Rozkład parametru {param} (Najlepsze przybliżenie)')
        plt.xlabel('Wartość parametru')
        plt.ylabel('Gęstość')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'EGARCH_BEST_{param}.png'))
        plt.close()

    # Zapis do Excela
    df_stat = pd.DataFrame(raport_stat)
    with pd.ExcelWriter(output_excel) as writer:
        df_ok.to_excel(writer, sheet_name='Parametry_EGARCH', index=False)
        df_stat.to_excel(writer, sheet_name='Testy_p_value', index=False)

    print(f"\nGOTOWE. Wyniki zapisano w: {output_excel}")
    print("\nPodsumowanie p-values (H0: dane pochodzą z danego rozkładu):")
    print(df_stat[['Parametr', 'p-val Normalny', 'p-val t-Student', 'p-val Log-normalny', 'p-val F']])

except Exception as e:
    print(f"Wystąpił błąd: {e}")
