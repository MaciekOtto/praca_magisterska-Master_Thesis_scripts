"""
stock_scraping.py - Pobieranie danych giełdowych (NASDAQ)

Pobiera dzienne ceny zamknięcia dla spółek z listy `nasdaq_top500.csv`
za pomocą biblioteki yfinance, dla okresu 2020-02-12 - 2025-11-12.
Spółki z niepełną historią (mniej niż 90% dni roboczych z danymi) są
odrzucane, a pozostałe uzupełniane metodą forward-fill względem pełnego
kalendarza dni roboczych.

Wejście: nasdaq_top500.csv (lista tickerów)
Wyjście: dane1000close.xlsx (ceny zamknięcia, kolumny = tickery)

----------------------------------------------------------------------

stock_scraping.py - Stock price data download (NASDAQ)

Downloads daily closing prices for the companies listed in
`nasdaq_top500.csv` via the yfinance library, for the period
2020-02-12 to 2025-11-12. Companies with incomplete history (less than
90% of business days with data) are dropped; the remaining series are
forward-filled against the full business-day calendar.

Input: nasdaq_top500.csv (ticker list)
Output: dane1000close.xlsx (closing prices, columns = tickers)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

# Wczytywania tickerów z CSV 
def load_tickers_from_csv(file_path='nasdaq_top500.csv', max_tickers=1721):
    try:
         
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        tickers = []
        for line in lines:
            ticker = line.strip().upper()  
            if ticker and ticker.isalpha() and 1 <= len(ticker) <= 5: 
                tickers.append(ticker)
        
        # Usuwanie duplikaty
        unique_tickers = list(dict.fromkeys(tickers))
        tickers = unique_tickers[:max_tickers]
        print(f"Załadowano {len(tickers)} poprawnych tickerów z pliku (zachowując kolejność): {tickers[:10]}...") 
        return tickers
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku CSV: {e}")
        return []

# Parametry
min_data_ratio = 0.9  

start_date = '2020-02-12'
end_date = '2025-11-13'  

tickers = load_tickers_from_csv(max_tickers=1721)
if not tickers:
    print("Brak tickerów – użyj przykładowej listy.")
    tickers = ['AAPL', 'MSFT', 'NVDA']  

print(f"Pobieranie danych dla {len(tickers)} spółek z okresu {start_date} do 2025-11-12...")
try:
    # Pobieranie pełnych danych 
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        print(f"Pobrano dane dla {len(data.columns.levels[1])} spółek z {len(tickers)} żądanych.")
    else:
        print(f"Pobrano dane dla 1 spółki.")
except Exception as e:
    print(f"Błąd podczas pobierania danych: {e}")
    data = pd.DataFrame()  

if data.empty:
    print("Brak danych – sprawdź tickery lub połączenie internetowe.")
    exit()

# Wyłącznie ceny zamknięcia (Close)
if isinstance(data.columns, pd.MultiIndex):
    close_data = data['Close']
else:
    close_data = pd.DataFrame({tickers[0]: data['Close']})

# Utwórz zakres wszystkich dni roboczych (5-dniowy tydzień: pon-pt)
all_business_days = pd.date_range(start=start_date, end='2025-11-12', freq='B')
total_business_days = len(all_business_days)

# Filtruj spółki na podstawie dostępności danych
filtered_tickers = []
excluded_tickers = []
for ticker in tickers:
    if ticker in close_data.columns:
        series = close_data[ticker].dropna()
        data_days = len(series)
        ratio = data_days / total_business_days
        if ratio >= min_data_ratio:
            filtered_tickers.append(ticker)
        else:
            excluded_tickers.append((ticker, ratio))

print(f"Po filtrze: {len(filtered_tickers)} spółek z pełnymi danymi (co najmniej {min_data_ratio*100}% dni). Wykluczonych: {len(excluded_tickers)}.")
if excluded_tickers:
    print(f"Wykluczone spółki (ticker, procent danych): {excluded_tickers[:10]}...")

filled_data = pd.DataFrame(index=all_business_days)
for ticker in filtered_tickers:
    # Reindeksowanie do pełnego kalendarza biznesowego i uzupełnienie braków poprzednią wartością (forward fill)
    filled_data[ticker] = close_data[ticker].reindex(all_business_days).ffill()

# Reset indeksu: data jako pierwsza kolumna
filled_data.reset_index(inplace=True)
filled_data.rename(columns={'index': 'Data'}, inplace=True)

# Zapisanie do Excel
output_file = 'dane1000close.xlsx'
filled_data.to_excel(output_file, index=False)
print(f"Dane zapisane do pliku {output_file}. Liczba kolumn (oprócz daty): {len(filled_data.columns) - 1}")
