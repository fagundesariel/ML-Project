from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/kidney_disease.csv")
INTERIM_DATA_PATH = Path("data/interim/kidney_disease_interim.csv")

def perform_minimal_data_cleaning():
    """Load raw kidney disease data, clean it minimally, and save as interim dataset."""
    
    print("Loading raw dataset...")

    df = pd.read_csv(
        RAW_DATA_PATH,
        header=0,
        skiprows=[1, 2],
        na_values=["?"]
    )

    print("Saving interim dataset...")

    INTERIM_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        INTERIM_DATA_PATH,
        index=False
    )

    print("Interim dataset saved to:")
    print(INTERIM_DATA_PATH)

if __name__ == "__main__":
    perform_minimal_data_cleaning()
