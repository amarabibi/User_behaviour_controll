import pandas as pd
import numpy as np
from pathlib import Path
import re

INPUT_FOLDER = Path("input_files")
OUTPUT_FOLDER = Path("output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SKEW_THRESHOLD = 1.0

def clean_col(col):
    col = str(col).strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    return col

def find_first_match(columns, candidates):
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    return None

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def apply_skewness_transform(df, threshold=SKEW_THRESHOLD):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report = []

    for col in numeric_cols:
        if col in ["trigger_alert", "high_stress_hr"]:
            continue

        s = df[col]
        skew_before = s.skew(skipna=True)

        if pd.isna(skew_before):
            continue

        if abs(skew_before) > threshold:
            if (s.dropna() >= 0).all():
                df[col] = np.log1p(s)
                method = "log1p"
            else:
                shifted = s - s.min() + 1
                df[col] = np.log1p(shifted)
                method = "shifted_log1p"

            skew_after = df[col].skew(skipna=True)
            report.append({
                "column": col,
                "skew_before": skew_before,
                "skew_after": skew_after,
                "method": method
            })

    return df, pd.DataFrame(report)

def feature_engineering(df, source_name):
    df.columns = [clean_col(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["source_file"] = source_name

    col_map = {
        "user_behavior_class": ["user_behavior_class", "behavior_class", "class", "label"],
        "stress_level": ["stress_level", "stress", "stressscore"],
        "heart_rate": ["heart_rate", "heartrate", "hr"],
        "app_usage_time": [
            "app_usage_time_minday",
            "app_usage_time_day",
            "app_usage_time",
            "screen_time_hoursday",
            "screen_on_time_hoursday",
        ],
        "student_id": ["student_id", "studentid", "user_id", "userid", "id"],
    }

    selected = {}
    for key, candidates in col_map.items():
        match = find_first_match(df.columns, candidates)
        if match is not None:
            selected[key] = match

    for new_name, old_name in selected.items():
        if new_name not in df.columns:
            df[new_name] = df[old_name]

    for col in ["stress_level", "heart_rate", "app_usage_time"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if "stress_level" in df.columns and "heart_rate" in df.columns:
        df["stress_hr_product"] = df["stress_level"] * df["heart_rate"]
        df["stress_hr_sum"] = df["stress_level"] + df["heart_rate"]
        df["stress_hr_diff"] = (df["stress_level"] - df["heart_rate"]).abs()
        df["stress_hr_ratio"] = df["stress_level"] / (df["heart_rate"].replace(0, np.nan))
    else:
        df["stress_hr_product"] = np.nan
        df["stress_hr_sum"] = np.nan
        df["stress_hr_diff"] = np.nan
        df["stress_hr_ratio"] = np.nan

    if "app_usage_time" in df.columns and "stress_level" in df.columns:
        df["app_stress_product"] = df["app_usage_time"] * df["stress_level"]
        df["app_stress_ratio"] = df["app_usage_time"] / (df["stress_level"].replace(0, np.nan))
    else:
        df["app_stress_product"] = np.nan
        df["app_stress_ratio"] = np.nan

    if "user_behavior_class" in df.columns:
        df["trigger_alert"] = df["user_behavior_class"].astype(str).str.strip().str.lower().isin(
            ["anomalous", "high stress"]
        )
        df["behavior_length"] = df["user_behavior_class"].astype(str).str.len()
    else:
        df["trigger_alert"] = False
        df["behavior_length"] = np.nan

    if "student_id" in df.columns:
        df["student_id_length"] = df["student_id"].astype(str).str.len()
    else:
        df["student_id_length"] = np.nan

    if "stress_level" in df.columns:
        df["stress_level_z"] = (df["stress_level"] - df["stress_level"].mean()) / df["stress_level"].std(ddof=0)

    if "heart_rate" in df.columns:
        df["heart_rate_z"] = (df["heart_rate"] - df["heart_rate"].mean()) / df["heart_rate"].std(ddof=0)

    if "app_usage_time" in df.columns:
        df["app_usage_time_z"] = (df["app_usage_time"] - df["app_usage_time"].mean()) / df["app_usage_time"].std(ddof=0)

    if "stress_level" in df.columns and "heart_rate" in df.columns:
        stress_threshold = df["stress_level"].quantile(0.75)
        heart_rate_threshold = df["heart_rate"].quantile(0.75)
        df["high_stress_hr"] = (
            (df["stress_level"] >= stress_threshold) &
            (df["heart_rate"] >= heart_rate_threshold)
        )
    else:
        df["high_stress_hr"] = False

    app_threshold = df["app_usage_time"].quantile(0.75) if "app_usage_time" in df.columns else None

    def make_action(row):
        if row.get("trigger_alert", False):
            if row.get("high_stress_hr", False):
                if app_threshold is not None and pd.notna(row.get("app_usage_time")) and row.get("app_usage_time", 0) >= app_threshold:
                    return "ALERT: High risk. Suggest break and lock social media apps."
                return "ALERT: High risk. Suggest break."
            return "ALERT: Behavior flagged."
        return "Normal"

    df["system_action"] = df.apply(make_action, axis=1)

    return df

def main():
    csv_files = sorted(INPUT_FOLDER.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in input_files/")
        return

    processed_parts = []
    alerts_parts = []
    summary_rows = []
    skew_reports = []

    for file in csv_files:
        raw_df = pd.read_csv(file, low_memory=False)
        processed_df = feature_engineering(raw_df, file.stem)
        processed_df, skew_report = apply_skewness_transform(processed_df)

        alerts_df = processed_df[processed_df["trigger_alert"] | processed_df["high_stress_hr"]].copy()

        processed_parts.append(processed_df)
        alerts_parts.append(alerts_df)

        summary_rows.append({
            "source_file": file.stem,
            "rows": len(processed_df),
            "alerts": len(alerts_df),
            "columns": len(processed_df.columns)
        })

        if not skew_report.empty:
            skew_report.insert(0, "source_file", file.stem)
            skew_reports.append(skew_report)

    cleaned_df = pd.concat(processed_parts, ignore_index=True, sort=False)
    alerts_df = pd.concat(alerts_parts, ignore_index=True, sort=False)

    cleaned_path = OUTPUT_FOLDER / "behavior_monitoring_output.csv"
    alerts_path = OUTPUT_FOLDER / "behavior_alerts_only.csv"
    summary_path = OUTPUT_FOLDER / "summary.csv"
    skew_path = OUTPUT_FOLDER / "skewness_report.csv"

    cleaned_df.to_csv(cleaned_path, index=False)
    alerts_df.to_csv(alerts_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    if skew_reports:
        pd.concat(skew_reports, ignore_index=True).to_csv(skew_path, index=False)
    else:
        pd.DataFrame(columns=["source_file", "column", "skew_before", "skew_after", "method"]).to_csv(skew_path, index=False)

    print(f"Saved: {cleaned_path}")
    print(f"Saved: {alerts_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {skew_path}")

if __name__ == "__main__":
    main()