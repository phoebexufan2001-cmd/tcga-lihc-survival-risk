"""Download the public TCGA-LIHC files used by the analysis."""

from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FILES = {
    "TCGA.LIHC.HiSeqV2.gz": (
        "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
        "TCGA.LIHC.sampleMap%2FHiSeqV2.gz"
    ),
    "LIHC_clinicalMatrix.tsv": (
        "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
        "TCGA.LIHC.sampleMap%2FLIHC_clinicalMatrix"
    ),
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        destination = RAW / filename
        if destination.exists():
            print(f"Already exists: {destination}")
            continue
        print(f"Downloading {filename}...")
        urlretrieve(url, destination)
    print("Data download complete.")


if __name__ == "__main__":
    main()

