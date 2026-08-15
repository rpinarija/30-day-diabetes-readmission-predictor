"""Download the UCI 'Diabetes 130-US hospitals' dataset into data/raw/."""

import io
import sys
import zipfile
from pathlib import Path

import requests

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED_FILES = ["diabetic_data.csv", "IDS_mapping.csv"]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / name).exists() for name in EXPECTED_FILES):
        print(f"Dataset already present in {RAW_DIR}")
        return

    response = requests.get(UCI_ZIP_URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        members = {Path(m).name: m for m in archive.namelist()}
        for name in EXPECTED_FILES:
            if name not in members:
                sys.exit(f"{name} not found in archive")
            (RAW_DIR / name).write_bytes(archive.read(members[name]))


if __name__ == "__main__":
    main()