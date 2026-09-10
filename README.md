# Modele GARCH a metody uczenia maszynowego w prognozowaniu zmienności - skrypty do pracy magisterskiej

Repozytorium zawiera kod źródłowy (Python) wykorzystany w empirycznej części pracy magisterskiej poświęconej porównaniu klasycznych modeli rodziny GARCH (GARCH, EGARCH, GJR-GARCH, APARCH) oraz metod uczenia maszynowego (Random Forest, LSTM, SVR) w prognozowaniu zmienności stóp zwrotu, w kontekście słabej formy hipotezy rynku efektywnego (EMH) oraz hipotezy rynku adaptacyjnego (AMH).

Próba badawcza obejmuje **1000 spółek notowanych na NASDAQ**, dla których wykorzystano dzienne ceny zamknięcia z okresu **12.02.2020 - 12.11.2025**, pobrane za pomocą biblioteki `yfinance`.

## Spis treści

- [Cel projektu](#cel-projektu)
- [Struktura repozytorium](#struktura-repozytorium)
- [Kolejność uruchamiania skryptów (pipeline)](#kolejność-uruchamiania-skryptów-pipeline)
- [Wymagania](#wymagania)
- [Dane wejściowe](#dane-wejściowe)
- [Znane ograniczenia / rzeczy do poprawy](#znane-ograniczenia--rzeczy-do-poprawy)

## Cel projektu

Głównym celem pracy jest weryfikacja, czy zmienność stóp zwrotu jest w ogóle przewidywalna (hipoteza zerowa: zmienność jest całkowicie nieprzewidywalna) oraz czy metody uczenia maszynowego systematycznie przewyższają klasyczne modele GARCH pod względem trafności prognoz out-of-sample. Dodatkowo, w oparciu o model CAPM oraz trzyczynnikowy model Famy-Frencha (FF3), sprawdzane jest, czy ekspozycja spółki na czynniki ryzyka rynkowego, wielkości (SMB) i wartości (HML) wyjaśnia zróżnicowanie błędów prognoz między spółkami.

## Struktura repozytorium

Repozytorium ma obecnie płaską strukturę (wszystkie skrypty w katalogu głównym). Poniżej pliki pogrupowane funkcjonalnie:

| Grupa | Pliki | Opis |
|---|---|---|
| **Pozyskanie danych** | `stock_scraping.py` | Pobiera dzienne ceny zamknięcia dla tickerów z `nasdaq_top500.csv` przez `yfinance`, filtruje spółki z niepełną historią, zapisuje `dane1000close.xlsx` |
| **Przygotowanie danych** | `log_returns.py` | Liczy logarytmiczne stopy zwrotu z `dane1000close.xlsx` → `dane1000stopy.xlsx` |
| **Testy wstępne** | `test_adf.py`, `ADF_vis.py`, `test_arch.py`, `arch_vis.py`, `test_wilska.py`, `desc_stats_and_vis.py` | Test ADF (stacjonarność), test ARCH (efekt ARCH na resztach ARIMA(1,0,1)), test Shapiro-Wilka (normalność), statystyki opisowe stóp zwrotu wraz z wizualizacjami |
| **Estymacja modeli GARCH-rodziny (pełna próba)** | `garch.py`, `egarch.py`, `gjrgarch.py`, `aparch.py` | Estymacja parametrów modeli GARCH(1,1), EGARCH, GJR-GARCH, APARCH dla każdej spółki + histogramy rozkładu parametrów |
| **Diagnostyka reszt modeli** | `rozklad_garch4testy.py`, `rozklad_egarch4testy.py`, `rozklad_gjr4testy.py`, `rozklad_aparch4testy.py` | Testy statystyczne na standaryzowanych resztach oszacowanych modeli GARCH-rodziny |
| **Prognozowanie out-of-sample (expanding window, firm-by-firm)** | `OOS_rolling_windowGARCH.py`, `RF_oos_rw.py`, `lstm_oos_rw.py`, `svr_oos_rw.py` | Prognozy zmienności metodą rozszerzającego się okna (`TRAIN_SIZE=1250`, okno testowe = 250 obserwacji) dla wszystkich sześciu modeli; wyniki jako pliki `.parquet` (prognozy) i `.xlsx` (RMSE/MAE); checkpointy zapisywane cyklicznie |
| **Naprawa formatu danych** | `fix_garchparquet.py` | Naprawia nazwy kolumn MultiIndex w pliku `garch_prognozy_oos.parquet` (problem z serializacją `np.str_`) |
| **Test Diebolda-Mariano** | `test_diebolda_mariano.py` | Porównuje trafność prognoz par modeli GARCH vs. ML (błąd kwadratowy) → `wyniki_diebold_mariano.xlsx` |
| **Porównanie błędów prognoz** | `error_vis.py` | Wykresy porównawcze RMSE/MAE dla wszystkich sześciu modeli |
| **Regresja przekrojowa CAPM / FF3** | `capm_3factors.py`, `3factors_R2_visuals.py`, `3factors_visuals_scatter.py` | Dwuetapowa regresja: (1) estymacja bet FF3/CAPM dla każdej spółki, (2) regresja przekrojowa RMSE względem bet; wizualizacje R² oraz zależności RMSE-bety |

## Kolejność uruchamiania skryptów (pipeline)

1. **`stock_scraping.py`** - pobiera dane cenowe (wymaga `nasdaq_top500.csv` w katalogu głównym) → `dane1000close.xlsx`
2. **`log_returns.py`** - liczy stopy zwrotu → `dane1000stopy.xlsx`
3. **Testy wstępne** (opcjonalnie, w dowolnej kolejności): `test_adf.py` → `ADF_vis.py`; `test_arch.py` → `arch_vis.py`; `test_wilska.py`; `desc_stats_and_vis.py`
4. **Estymacja GARCH-rodziny na pełnej próbie**: `garch.py`, `egarch.py`, `gjrgarch.py`, `aparch.py`, następnie odpowiadające im `rozklad_*4testy.py`
5. **Prognozowanie out-of-sample**: `OOS_rolling_windowGARCH.py` oraz `RF_oos_rw.py`, `ltsm_oos_rw.py`, `svr_oos_rw.py` (można uruchamiać niezależnie/równolegle - każdy zapisuje własne checkpointy i pliki wynikowe)
6. Jeżeli pojawi się błąd dotyczący formatu kolumn w pliku parquet z prognozami GARCH → uruchom **`fix_garchparquet.py`**
7. **`test_diebolda_mariano.py`** - wymaga plików `.parquet` z kroku 5 (dla GARCH-rodziny: wersji `_fixed`, patrz krok 6)
8. **`error_vis.py`** - wymaga plików `*_rmse_mae.xlsx` z kroku 5
9. **`capm_3factors.py`** - wymaga `dane1000stopy.xlsx`, pliku czynników Famy-Frencha `F-F_Research_Data_Factors_daily.txt` (do pobrania ze strony Kennetha Frencha, patrz sekcja *Dane wejściowe*) oraz plików `*_rmse_mae.xlsx`
10. **`3factors_R2_visuals.py`** i **`3factors_visuals_scatter.py`** - ⚠️ wymagają ręcznego przygotowania pośrednich plików (`ff3_regresja_przekrojowa_2.xlsx`, `ff3_regresja_przekrojowa_vis.xlsx`) na podstawie konkretnych arkuszy z pliku wyjściowego kroku 9 - patrz sekcja *Znane ograniczenia*.

## Wymagania

Skrypty testowane z Pythonem 3.11.0. Kluczowe zależności (pełna lista z przypiętymi wersjami w `requirements.txt`):

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
arch
scikit-learn
tensorflow
yfinance
pyarrow
openpyxl
```

Zalecane utworzenie osobnego środowiska wirtualnego i instalacja przez:

```bash
pip install -r requirements.txt
```

## Dane wejściowe

- `nasdaq_top500.csv` - lista tickerów wejściowych, znajduje się w repozytorium.
- `dane1000close.xlsx`, `dane1000stopy.xlsx` - pliki generowane lokalnie przez `stock_scraping.py` i `log_returns.py`; **nie są dołączone do repozytorium** (rozmiar / dane pobierane na bieżąco z Yahoo Finance).
- `F-F_Research_Data_Factors_daily.txt` - dzienne czynniki Famy-Frencha (Mkt-RF, SMB, HML, RF), pobierane ze strony [Kennetha Frencha (Dartmouth)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html); 

## Znane ograniczenia / rzeczy do poprawy

- Ścieżki plików wejściowych/wyjściowych są zapisane na sztywno w kodzie (brak `argparse`/pliku konfiguracyjnego).
- Pliki `*_blankvisuals.py` to wcześniejsze/alternatywne wersje głównych skryptów estymacji GARCH-rodziny (bez części diagnostycznej) - pozostawione w repozytorium jako punkt odniesienia, docelowo do uporządkowania.
- Kroki 10. w pipeline (`3factors_R2_visuals.py`, `3factors_visuals_scatter.py`) wymagają obecnie ręcznego wydzielenia odpowiednich arkuszy z pliku `ff3_regresja_przekrojowa.xlsx` do osobnych plików - planowana automatyzacja tego kroku.

