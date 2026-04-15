from pathlib import Path
import pandas as pd
import re

input_folder = Path("input_files")
output_folder = Path("output")
output_folder.mkdir(parents=True, exist_ok=True)

csv_files = sorted(input_folder.glob("*.csv"))

def clean_col(col):
    col = str(col).strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    return col

if not csv_files:
    print("No CSV files found in the input folder.")
else:
    print(f"Processing {len(csv_files)} files...")
    print("Combining datasets...")

    dfs = []
    file_names = []

    for file in csv_files:
        df = pd.read_csv(file, low_memory=False)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # standardize column names
        df.columns = [clean_col(c) for c in df.columns]

        # remove duplicate columns again after renaming
        df = df.loc[:, ~df.columns.duplicated()].copy()

        df["source_file"] = file.stem
        dfs.append(df)
        file_names.append(file.stem)

    combined_df = pd.concat(dfs, ignore_index=True, sort=False)

    output_path = output_folder / "combined_dataset.csv"
    combined_df.to_csv(output_path, index=False)

    with open(output_folder / "source_files.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(file_names))

    print(f"✅ Saved: {output_path}")