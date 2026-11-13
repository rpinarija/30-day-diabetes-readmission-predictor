"""Download the UCI 'Diabetes 130-US hospitals' dataset into data/raw/.

Each file is checked against a pinned SHA-256 fingerprint, so a changed or corrupted
upstream file stops the pipeline instead of silently training on different data.
"""

import hashlib
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

# Fingerprints of the exact files the current model was trained on.
EXPECTED_SHA256 = {
    "diabetic_data.csv": "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
    "IDS_mapping.csv": "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(name: str, data: bytes) -> None:
    actual = sha256(data)
    if actual != EXPECTED_SHA256[name]:
        sys.exit(
            f"{name} does not match the pinned dataset (sha256 {actual}). "
            "The upstream file changed or is corrupted; investigate before training."
        )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / name).exists() for name in EXPECTED_SHA256):
        for name in EXPECTED_SHA256:
            verify(name, (RAW_DIR / name).read_bytes())
        print(f"Dataset already present and verified in {RAW_DIR}")
        return

    response = requests.get(UCI_ZIP_URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        members = {Path(m).name: m for m in archive.namelist()}
        for name in EXPECTED_SHA256:
            if name not in members:
                sys.exit(f"{name} not found in archive")
            data = archive.read(members[name])
            verify(name, data)
            (RAW_DIR / name).write_bytes(data)
    print(f"Downloaded and verified dataset in {RAW_DIR}")


if __name__ == "__main__":
    main()
