from pathlib import Path
import pandas as pd
import numpy as np


def clean_dataset(
    df,
    id_cols=None,
    date_cols=None,
    numeric_cols=None,
    binary_cols=None,
    categorical_cols=None,
    text_cols=None,
    outlier_cols=None,
    drop_missing_threshold=0.7
):
    df = df.copy()

    id_cols = id_cols or []
    date_cols = date_cols or []
    numeric_cols = numeric_cols or []
    binary_cols = binary_cols or []
    categorical_cols = categorical_cols or []
    text_cols = text_cols or []
    outlier_cols = outlier_cols or []

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )

    df = df.drop_duplicates()
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[^0-9.\-]", "", regex=True)
                .replace("", np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    binary_map = {
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "0": 0
    }
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(binary_map)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan, "none": np.nan, "": np.nan})
            )

    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .replace({"nan": np.nan, "none": np.nan, "": np.nan})
            )

    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isna().mean() < drop_missing_threshold:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].isna().mean() < drop_missing_threshold:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

    for col in outlier_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower, upper)

    return df


input_folder = Path("input_files")
output_folder = Path("cleaned_files")
output_folder.mkdir(exist_ok=True)

for file_path in input_folder.glob("*.csv"):
    df = pd.read_csv(file_path, low_memory=False)

    cleaned = clean_dataset(
        df,
        id_cols=["customer_id", "order_id"],
        date_cols=["order_date", "delivery_date", "timestamp", "login_time"],
        numeric_cols=["age", "salary", "amount", "quantity"],
        binary_cols=["is_active", "paid", "verified", "admin_notified", "webcam_verification"],
        categorical_cols=["gender", "city", "status", "category", "device_used", "update_type"],
        text_cols=["name", "description", "comments", "email", "user_name", "anomaly type", "mitigation action"],
        outlier_cols=["salary", "amount", "quantity", "age"]
    )

    out_path = output_folder / f"{file_path.stem}_cleaned.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")