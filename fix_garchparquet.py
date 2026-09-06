"""
fix_garchparquet.py - Naprawa formatu kolumn w pliku prognoz GARCH

Narzędzie pomocnicze uruchamiane, gdy zapis wielopoziomowych kolumn
(MultiIndex: model, ticker) w pliku parquet z prognozami GARCH-rodziny
zostanie zserializowany niepoprawnie (jako tekstowe reprezentacje
krotek `np.str_(...)` zamiast właściwego MultiIndex). Odczytuje plik,
odtwarza poprawny MultiIndex kolumn i zapisuje naprawioną wersję.

Wejście: garch_prognozy_oos.parquet
Wyjście: garch_prognozy_oos_fixed.parquet

----------------------------------------------------------------------

fix_garchparquet.py - Fixes column format in the GARCH forecast file

Helper script run when the multi-level columns (MultiIndex: model,
ticker) of the GARCH-family forecast parquet file get serialized
incorrectly (as text representations of tuples, `np.str_(...)`,
instead of a proper MultiIndex). Reads the file, reconstructs the
correct column MultiIndex, and writes the fixed version.

Input: garch_prognozy_oos.parquet
Output: garch_prognozy_oos_fixed.parquet
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import ast

# 1. odczyt
table = pq.read_table(
    "garch_prognozy_oos.parquet",
    use_pandas_metadata=False
)

df = table.to_pandas(ignore_metadata=True)


# 2. naprawa kolumn
new_cols = []

for c in df.columns:

    if isinstance(c, tuple):
        model = str(c[0])
        ticker = str(c[1])

    else:
        txt = str(c)

        txt = txt.replace("np.str_(", "")
        txt = txt.replace(")", "")

        try:
            parsed = ast.literal_eval(txt)

            if isinstance(parsed, tuple):
                model = str(parsed[0])
                ticker = str(parsed[1])

            else:
                model = txt
                ticker = ""

        except:
            model = txt
            ticker = ""

    new_cols.append((model, ticker))


df.columns = pd.MultiIndex.from_tuples(new_cols)


# 3. indeksowanie
if "index" in df.columns:
    df = df.set_index("index")

df.index = pd.to_datetime(df.index)


print("Pierwsze kolumny po naprawie:")
print(df.columns[:5])


# 4. zapis
table_fixed = pa.Table.from_pandas(
    df,
    preserve_index=True
)

table_fixed = table_fixed.replace_schema_metadata(None)

pq.write_table(
    table_fixed,
    "garch_prognozy_oos_fixed.parquet"
)


print("Gotowe")
