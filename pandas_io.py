"""
=============================================================
  PANDAS IMPORT / EXPORT — Standalone Lab Program
=============================================================
  Covers: read_csv, head, tail, describe, info, shape,
          dtypes, to_csv, to_excel
  Libraries: pandas
  Usage: python pandas_io.py
=============================================================
"""

import pandas as pd


def run():
    # --- Load Dataset ---
    path = input("Enter CSV file path (default: data.csv): ").strip() or "data.csv"
    df = pd.read_csv(path)

    print(f"\n✅ Dataset loaded from: {path}")
    print(f"\n--- Shape ---\n  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"\n--- Data Types ---\n{df.dtypes}")
    print(f"\n--- First 5 Rows ---\n{df.head()}")
    print(f"\n--- Last 5 Rows ---\n{df.tail()}")
    print(f"\n--- Statistical Summary ---\n{df.describe()}")
    print(f"\n--- Info ---")
    df.info()
    print(f"\n--- Missing Values ---\n{df.isnull().sum()}")

    # --- Export ---
    export = input("\nExport data? (csv / excel / skip): ").strip().lower()
    if export == 'csv':
        out = input("Output filename (default: output.csv): ").strip() or "output.csv"
        df.to_csv(out, index=False)
        print(f"✅ Exported to {out}")
    elif export == 'excel':
        out = input("Output filename (default: output.xlsx): ").strip() or "output.xlsx"
        df.to_excel(out, index=False)
        print(f"✅ Exported to {out}")
    else:
        print("Skipped export.")


if __name__ == "__main__":
    run()
