from pathlib import Path
import shutil

import kagglehub


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Kaggle dataset
DATASET_HANDLE = "blastchar/telco-customer-churn"
SOURCE_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_FILENAME = "telco_customer_churn.csv"


def download_dataset() -> Path:
    """Download the Telco Customer Churn dataset into data/raw."""

    print("Starting dataset download...")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # KaggleHub downloads/caches the dataset and returns its location
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))

    source_file = dataset_path / SOURCE_FILENAME

    if not source_file.exists():
        raise FileNotFoundError(
            f"Expected dataset file was not found: {source_file}"
        )

    destination = RAW_DATA_DIR / OUTPUT_FILENAME

    shutil.copy2(source_file, destination)

    print("Dataset download complete.")
    print(f"Source: {source_file}")
    print(f"Saved to: {destination}")

    return destination


if __name__ == "__main__":
    download_dataset()