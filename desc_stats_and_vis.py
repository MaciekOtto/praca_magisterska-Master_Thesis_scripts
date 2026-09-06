"""
desc_stats_and_vis.py - Statystyki opisowe stóp zwrotu

Liczy statystyki opisowe (średnia, odchylenie standardowe, wariancja,
skośność, kurtoza) stóp zwrotu dla każdej spółki i rysuje histogramy
ich rozkładów (po winsoryzacji na poziomie 1%/99%, wyłącznie do celów
wizualizacji).

Wejście: dane1000stopy.xlsx
Wyjście: statystyki_opisowe.xlsx, histogramy_statystyk.png

----------------------------------------------------------------------

desc_stats_and_vis.py - Descriptive statistics of returns

Computes descriptive statistics (mean, standard deviation, variance,
skewness, kurtosis) of returns for each company and plots histograms
of their distributions (winsorized at the 1%/99% level, for
visualization purposes only).

Input: dane1000stopy.xlsx
Output: statystyki_opisowe.xlsx, histogramy_statystyk.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

INPUT_FILE = "dane1000stopy.xlsx"
OUTPUT_XLSX = "statystyki_opisowe.xlsx"
OUTPUT_PLOT = "histogramy_statystyk.png"

# wczytanie danych
df = pd.read_excel(INPUT_FILE, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce")

# statystyki
stats = pd.DataFrame({
    "srednia": df.mean(),
    "odch_std": df.std(),
    "wariancja": df.var(),
    "skosnosc": df.apply(lambda x: skew(x.dropna())),
    "kurtoza": df.apply(lambda x: kurtosis(x.dropna())), 
})

stats.index.name = "spolka"
stats.to_excel(OUTPUT_XLSX)

# winsoryzacja
def winsorize(series, lower=0.01, upper=0.99):
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)

stats_wins = stats.apply(winsorize)

# Histogramy
bins_map = {
    "srednia": 40,
    "odch_std": 40,
    "wariancja": 50,
    "skosnosc": 40,
    "kurtoza": 30,
}

titles = {
    "srednia": "Rozkład średnich dziennych cen zamknięcia",
    "odch_std": "Rozkład odchyleń standardowych",
    "wariancja": "Rozkład wariancji",
    "skosnosc": "Rozkład skośności",
    "kurtoza": "Rozkład kurtozy",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, (col, bins) in zip(axes, bins_map.items()):
    ax.hist(stats_wins[col], bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(titles[col], fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel("Liczba spółek")

fig.delaxes(axes[-1])

fig.suptitle("Rozkłady statystyk opisowych cen zamknięcia\n"
              , fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_PLOT, dpi=150)

print("Zapisano:", OUTPUT_XLSX, "oraz", OUTPUT_PLOT)
print(stats.describe())
