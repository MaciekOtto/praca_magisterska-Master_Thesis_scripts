"""
error_vis.py - Porównanie błędów prognoz (RMSE/MAE) wszystkich modeli

Wczytuje wyniki RMSE/MAE ze wszystkich sześciu modeli (GARCH,
GJR-GARCH, EGARCH, APARCH, RF, LSTM, SVR) i tworzy wykresy porównawcze
(m.in. rozkłady błędów w podziale na model), pozwalające ocenić, który
model ma najniższe/najwyższe błędy prognoz w przekroju spółek.

Wejście: garch_rmse_mae.xlsx, rf_rmse_mae.xlsx, lstm_rmse_mae.xlsx,
         svr_rmse_mae.xlsx
Wyjście: wykresy_porownanie/*.png

----------------------------------------------------------------------

error_vis.py - Forecast error (RMSE/MAE) comparison across models

Loads RMSE/MAE results from all six models (GARCH, GJR-GARCH, EGARCH,
APARCH, RF, LSTM, SVR) and produces comparative charts (e.g., error
distributions broken down by model), allowing an assessment of which
model has the lowest/highest forecast errors across companies.

Input: garch_rmse_mae.xlsx, rf_rmse_mae.xlsx, lstm_rmse_mae.xlsx,
       svr_rmse_mae.xlsx
Output: wykresy_porownanie/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Ustawienia ──
OUTPUT_FOLDER = 'wykresy_porownanie'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GARCH_MODELS_TO_SHOW = ['GARCH', 'GJR-GARCH', 'EGARCH', 'APARCH']  

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ── Wczytanie i scalenie danych 

def load_all_rmse():

    frames = []

    df_garch = pd.read_excel('garch_rmse_mae.xlsx', sheet_name='Surowe')
    df_garch = df_garch[df_garch['Model'].isin(GARCH_MODELS_TO_SHOW)]
    frames.append(df_garch[['Spółka', 'Model', 'RMSE', 'MAE']])

    for model_name, filepath in [
        ('RF',   'rf_rmse_mae.xlsx'),
        ('LSTM', 'lstm_rmse_mae.xlsx'),
        ('SVR',  'svr_rmse_mae.xlsx'),
    ]:
        try:
            df = pd.read_excel(filepath, sheet_name='Surowe')
            if 'Model' not in df.columns:
                df['Model'] = model_name
            df = df[df['Model'] == model_name] if df['Model'].nunique() > 1 else df
            df['Model'] = model_name  
            frames.append(df[['Spółka', 'Model', 'RMSE', 'MAE']])
        except Exception as e:
            print(f"Uwaga: nie wczytano {filepath}: {e}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.dropna(subset=['RMSE'])

    # Usuwanie duplikatow
    df_all = df_all.drop_duplicates(subset=['Spółka', 'Model'])

    print(f"Wczytano {len(df_all)} rekordów:")
    print(df_all.groupby('Model')['RMSE'].agg(['count', 'median', 'mean']).round(6))

    return df_all


def winsorize(series, lower=0.01, upper=0.99):
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


# ── Wykres 1: Boxplot wszystkich modeli ────

def plot_boxplot_all(df_all):
    df_viz = df_all.copy()
    df_viz['RMSE_wins'] = df_viz.groupby('Model')['RMSE'].transform(winsorize)

    order = (df_viz.groupby('Model')['RMSE_wins']
             .median().sort_values().index.tolist())

    plt.figure(figsize=(11, 6.5))
    ax = sns.boxplot(data=df_viz, x='Model', y='RMSE_wins', order=order,
                     palette='Set2', showfliers=False)
    sns.stripplot(data=df_viz, x='Model', y='RMSE_wins', order=order,
                  color='black', alpha=0.15, size=2, jitter=0.25, ax=ax)

    plt.title('Rozkład RMSE prognoz zmienności — porównanie modeli\n',
              fontsize=11)
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '01_boxplot_RMSE_wszystkie_modele.png'))
    plt.close()
    print("Zapisano: 01_boxplot_RMSE_wszystkie_modele.png")


# ── Wykresy 2-4: Histogramy per model ML ──────────────────────────────────────

def plot_histogram_single(df_all, model_name, filename, color):
    data = df_all[df_all['Model'] == model_name]['RMSE'].dropna()
    data_wins = winsorize(data)

    plt.figure(figsize=(9, 5.5))
    sns.histplot(data_wins, bins=40, kde=True, color=color)
    plt.axvline(data.median(), color='red', linestyle='--',
               label=f'Mediana = {data.median():.6f}')
    plt.title(f'Rozkład RMSE prognoz zmienności — model {model_name}\n'
             f'(N = {len(data)} spółek)',
             fontsize=11)
    plt.xlabel('RMSE')
    plt.ylabel('Liczba spółek')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, filename))
    plt.close()
    print(f"Zapisano: {filename}")


# ── Wykres 5: Panel 3 modeli ML razem ──

def plot_panel_ml(df_all):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'RF': 'forestgreen', 'LSTM': 'steelblue', 'SVR': 'darkorange'}

    for ax, model_name in zip(axes, ['RF', 'LSTM', 'SVR']):
        data = df_all[df_all['Model'] == model_name]['RMSE'].dropna()
        data_wins = winsorize(data)
        sns.histplot(data_wins, bins=30, kde=True, color=colors[model_name], ax=ax)
        ax.axvline(data.median(), color='red', linestyle='--', linewidth=1)
        ax.set_title(f'{model_name}\nmediana={data.median():.5f}', fontsize=10)
        ax.set_xlabel('RMSE')
        ax.set_ylabel('Liczba spółek' if model_name == 'RF' else '')

    plt.suptitle('Rozkład RMSE — metody uczenia maszynowego',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '05_histogram_RMSE_panel_ML.png'))
    plt.close()
    print("Zapisano: 05_histogram_RMSE_panel_ML.png")


def export_summary_table(df_all):
    summary = (df_all.groupby('Model')['RMSE']
               .agg(['count', 'median', 'mean', 'std', 'min', 'max'])
               .round(6)
               .sort_values('median'))
    summary.columns = ['N', 'Mediana', 'Średnia', 'Odch.std', 'Min', 'Max']

    summary_mae = (df_all.groupby('Model')['MAE']
                  .agg(['median', 'mean', 'std'])
                  .round(6)
                  .sort_values('median'))
    summary_mae.columns = ['Mediana_MAE', 'Średnia_MAE', 'Odch.std_MAE']

    full_summary = summary.join(summary_mae)
    full_summary.to_excel(os.path.join(OUTPUT_FOLDER, 'tabela_podsumowujaca.xlsx'))
    print("\nTabela podsumowująca:")
    print(full_summary)
    print(f"\nZapisano: tabela_podsumowujaca.xlsx")


# ── MAIN ───

if __name__ == '__main__':
    df_all = load_all_rmse()

    plot_boxplot_all(df_all)
    plot_histogram_single(df_all, 'RF',   '02_histogram_RMSE_RF.png',   'forestgreen')
    plot_histogram_single(df_all, 'LSTM', '03_histogram_RMSE_LSTM.png', 'steelblue')
    plot_histogram_single(df_all, 'SVR',  '04_histogram_RMSE_SVR.png',  'darkorange')
    plot_panel_ml(df_all)
    export_summary_table(df_all)

    print(f"\nGOTOWE. Wszystkie wykresy w folderze: {OUTPUT_FOLDER}/")
