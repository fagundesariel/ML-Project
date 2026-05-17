from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/kidney_disease.csv")
INTERIM_DATA_PATH = Path("data/interim/kidney_disease_interim.csv")

def perform_minimal_data_cleaning():
    """Load raw kidney disease data, clean it minimally, and save as interim dataset."""

    print("Loading raw dataset...")

    # Read the CSV file, skipping the first two rows which contain metadata
    df = pd.read_csv(
        RAW_DATA_PATH,
        header=0,
        skiprows=[1, 2]
    )

    # Remove the row from the 'grf' column containing the value 'p',
    # avoiding making assumptions about what 'p' represents in the context of the dataset,
    # as it is unclear what 'p' stands for.
    df = df[df['grf'].str.strip() != 'p']

    # Remove columns that represent data leakage:
    # - 'affected': mirrors the target class (CKD diagnosis)
    # - 'stage': carries post-diagnosis information
    leakage_columns = ['affected', 'stage']
    df = df.drop(columns=[col for col in leakage_columns if col in df.columns])

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
