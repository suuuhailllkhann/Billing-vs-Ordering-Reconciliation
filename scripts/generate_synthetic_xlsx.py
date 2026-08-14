"""Regenerate the fictional XLSX ingestion fixture from its reviewed CSV source."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data" / "synthetic" / "billing_export_clean_aliases.csv"
DESTINATION = ROOT / "data" / "synthetic" / "billing_export_clean_aliases.xlsx"


def main() -> None:
    frame = pd.read_csv(SOURCE, dtype="string", keep_default_na=True)
    frame.to_excel(DESTINATION, index=False, engine="openpyxl")


if __name__ == "__main__":
    main()
