"""
test_wilska.py - Test normalności Shapiro-Wilka

Dla każdej spółki przeprowadza test Shapiro-Wilka na cenach zamknięcia
(pełny okres), aby sprawdzić normalność rozkładu - H0: rozkład
normalny (p > 0.05 = brak podstaw do odrzucenia). Generuje też wykres
słupkowy podsumowujący liczbę spółek z rozkładem normalnym / bez.

Wejście: dane1000close.xlsx
Wyjście: wyniki_shapiro_pelny_okres.xlsx,
         podsumowanie_shapiro_pelny_okres.png

----------------------------------------------------------------------

test_wilska.py - Shapiro-Wilk normality test

Runs the Shapiro-Wilk test on each company's closing-price series
(full period) to test for normality - H0: normal distribution
(p > 0.05 = fail to reject). Also produces a bar chart summarizing the
number of companies with / without a normal distribution.

Input: dane1000close.xlsx
Output: wyniki_shapiro_pelny_okres.xlsx,
        podsumowanie_shapiro_pelny_okres.png
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ustawienia ---
input_file = 'dane1000close.xlsx'
output_file = 'wyniki_shapiro_pelny_okres.xlsx'

print(f"Wczytanie danych z pliku: {input_file}...")

try:
    df = pd.read_excel(input_file)
    
    # Obsługa daty i czyszczenie
    col_data = next((c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()), None)
    if col_data:
        df.set_index(col_data, inplace=True)
    
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    df.dropna(inplace=True)

    print(f"Przeprowadzenie testu Shapiro-Wilka dla {len(df.columns)} spółek (pełny zakres)...")

    wyniki = []
    for ticker in df.columns:
        series = df[ticker].dropna()
        
        # Test Shapiro-Wilksa
        stat, p_val = shapiro(series)
        
        wyniki.append({
            'Spółka': ticker,
            'Statystyka W': round(stat, 6),
            'p-value': round(p_val, 10),
            'Czy Rozkład Normalny (p>0.05)': 'TAK' if p_val > 0.05 else 'NIE'
        })

    # Zapis wyników
    df_wyniki = pd.DataFrame(wyniki)
    df_wyniki.to_excel(output_file, index=False)

    # --- Podsumowanie wizualne ---
    counts = df_wyniki['Czy Rozkład Normalny (p>0.05)'].value_counts()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=counts.index, y=counts.values, palette=['red', 'green'])
    plt.title('Wyniki testu normalności Shapiro-Wilka (N=1500)')
    plt.ylabel('Liczba spółek')
    plt.xlabel('Czy rozkład jest normalny?')
    
    for i, val in enumerate(counts.values):
        plt.text(i, val, str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.savefig('podsumowanie_shapiro_pelny_okres.png', dpi=300)
    print(f"Wyniki zapisano w: {output_file}")
    print("Wykres podsumowujący zapisano jako: podsumowanie_shapiro_pelny_okres.png")
    
    # Wyświetlenie wniosku 
    if 'NIE' in counts.index:
        proc_nie = (counts['NIE'] / len(df.columns)) * 100
        print(f"\nUWAGA: {proc_nie:.2f}% spółek NIE posiada rozkładu normalnego.")

except Exception as e:
    print(f"Wystąpił błąd: {e}")
