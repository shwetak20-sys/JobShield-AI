
import html
import re
import json
from pathlib import Path

import pandas as pd


# -----------------------------------------
# 1. Project paths
# -----------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "fake_job_postings.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "cleaned_job_postings.csv"
REPORT_FILE = PROJECT_DIR / "data" / "preprocessing_report.json"


# -----------------------------------------
# 2. Text columns used by the ML model
# -----------------------------------------

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "department",
    "industry",
    "employment_type",
    "required_experience",
    "required_education",
    "function",
    "location",
]

TARGET_COLUMN = "fraudulent"


# -----------------------------------------
# 3. Text cleaning function
# -----------------------------------------

def clean_text(value):
    """Clean HTML entities, tags and unnecessary spaces."""

    if pd.isna(value):
        return ""

    value = str(value)
    value = html.unescape(value)

    # Remove HTML tags
    value = re.sub(r"<[^>]+>", " ", value)

    # Remove extra whitespace and newlines
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# -----------------------------------------
# 4. Main preprocessing function
# -----------------------------------------

def main():

    print("\nJOBSHIELD AI - DATA PREPROCESSING")
    print("=" * 45)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    # Load original dataset
    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)

    print(f"Original records: {original_rows}")
    print(f"Original columns: {len(df.columns)}")

    # Verify required columns
    required_columns = TEXT_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    # Validate target labels
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    invalid_labels = (
        df[TARGET_COLUMN].isna()
        | ~df[TARGET_COLUMN].isin([0, 1])
    )

    invalid_label_count = int(invalid_labels.sum())

    df = df.loc[~invalid_labels].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Clean all selected text columns
    for column in TEXT_COLUMNS:
        df[column] = df[column].apply(clean_text)

    # Build combined text using labelled sections
    def combine_fields(row):

        sections = []

        for column in TEXT_COLUMNS:

            value = row[column]

            if value:
                label = column.replace("_", " ").title()
                sections.append(f"{label}: {value}")

        return " ".join(sections)

    df["combined_text"] = df.apply(
        combine_fields, axis=1
    )

    # Remove records with no usable text
    empty_text = df["combined_text"].str.strip().eq("")
    empty_text_count = int(empty_text.sum())

    df = df.loc[~empty_text].copy()

    # Keep original columns and add combined_text
    df = df.reset_index(drop=True)

    # Dataset statistics
    final_rows = len(df)

    label_counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    missing_before = pd.read_csv(INPUT_FILE).isnull().sum()

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # Generate preprocessing report
    report = {
        "original_records": int(original_rows),
        "final_records": int(final_rows),
        "removed_invalid_labels": invalid_label_count,
        "removed_empty_text_records": empty_text_count,
        "exact_duplicate_rows_in_cleaned_data": int(
            df.duplicated().sum()
        ),
        "target_distribution": {
            str(key): int(value)
            for key, value in label_counts.items()
        },
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "text_columns_used": TEXT_COLUMNS,
        "missing_values_in_original_data": {
            key: int(value)
            for key, value in missing_before.items()
            if value > 0
        },
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    # Display results
    print("\nPREPROCESSING RESULTS")
    print("-" * 45)

    print(f"Original records: {original_rows}")
    print(f"Final records: {final_rows}")
    print(f"Invalid labels removed: {invalid_label_count}")
    print(f"Empty text records removed: {empty_text_count}")

    print("\nTARGET DISTRIBUTION:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    print("\nOUTPUT FILES:")
    print(OUTPUT_FILE)
    print(REPORT_FILE)

    print("\nSAMPLE CLEANED TEXT:")
    print(df["combined_text"].iloc[0][:500])

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
