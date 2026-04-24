from pathlib import Path
import pandas as pd
import re

input_file = Path("merged_file/combined_dataset.csv")
output_file = Path("extracted_file/combined_features.csv")

df = pd.read_csv(input_file, low_memory=False)

def clean_col(col):
    col = str(col).strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    return col

df.columns = [clean_col(c) for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()].copy()

if "source_file" not in df.columns:
    df["source_file"] = "unknown"

date_cols = [c for c in df.columns if any(k in c for k in ["date", "time", "timestamp"])]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_hour"] = df[col].dt.hour
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek

text_cols = [c for c in df.select_dtypes(include="object").columns if c != "source_file" and c not in date_cols]

for col in text_cols:
    df[f"{col}_len"] = df[col].astype(str).fillna("").str.len()
    df[f"{col}_words"] = df[col].astype(str).fillna("").str.split().str.len()

cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in date_cols]
df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

df = df.drop(columns=date_cols, errors="ignore")

important_cols = [c for c in df.columns if (
    c == "source_file" or
    c.endswith("_year") or
    c.endswith("_month") or
    c.endswith("_day") or
    c.endswith("_hour") or
    c.endswith("_dayofweek") or
    c.endswith("_len") or
    c.endswith("_words") or
    pd.api.types.is_numeric_dtype(df[c])
)]

df = df[important_cols]

df.to_csv(output_file, index=False)
print(f"Saved: {output_file}")