import streamlit as st
import pandas as pd
from pathlib import Path
import re

st.set_page_config(page_title="Behavior Monitoring Dashboard", layout="wide")
st.title("Behavior Monitoring Dashboard")

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

folder_path = st.text_input("input_files", value="input_files")

if st.button("Process CSV Files"):
    folder = Path(folder_path)

    if not folder.exists():
        st.error("Folder does not exist.")
    else:
        csv_files = sorted(folder.glob("*.csv"))

        if not csv_files:
            st.error("No CSV files found in the folder.")
        else:
            dfs = []
            source_files = []

            for file in csv_files:
                df = pd.read_csv(file, low_memory=False)
                df.columns = [clean_col(c) for c in df.columns]
                df = df.loc[:, ~df.columns.duplicated()].copy()
                df["source_file"] = file.stem
                dfs.append(df)
                source_files.append(file.stem)

            combined_df = pd.concat(dfs, ignore_index=True, sort=False)
            combined_df.columns = [clean_col(c) for c in combined_df.columns]

            col_map = {
                "user_behavior_class": ["user_behavior_class", "behavior_class", "class", "label"],
                "stress_level": ["stress_level", "stress", "stressscore"],
                "heart_rate": ["heart_rate", "heartrate", "hr"],
                "app_usage_time": ["app_usage_time_minday", "app_usage_time_day", "app_usage_time", "screen_time_hoursday", "screen_on_time_hoursday"],
                "student_id": ["student_id", "studentid", "user_id", "userid", "id"],
            }

            selected = {}
            for key, candidates in col_map.items():
                match = find_first_match(combined_df.columns, candidates)
                if match is not None:
                    selected[key] = match

            keep_cols = list(selected.values())
            if keep_cols:
                df = combined_df[keep_cols].copy()
            else:
                df = combined_df.copy()

            df = df.rename(columns={v: k for k, v in selected.items()})

            for col in ["stress_level", "heart_rate", "app_usage_time"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "user_behavior_class" in df.columns:
                df["trigger_alert"] = df["user_behavior_class"].astype(str).str.strip().str.lower().isin(
                    ["anomalous", "high stress"]
                )
            else:
                df["trigger_alert"] = False

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

            alerts_df = df[df["trigger_alert"] | df["high_stress_hr"]].copy()

            st.subheader("Combined and Filtered Data")
            st.dataframe(df, use_container_width=True)

            st.subheader("Alerts Only")
            st.dataframe(alerts_df, use_container_width=True)

            output_folder = Path("output")
            output_folder.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_folder / "behavior_monitoring_output.csv", index=False)
            alerts_df.to_csv(output_folder / "behavior_alerts_only.csv", index=False)

            st.success(f"Processed {len(csv_files)} files.")
            st.write(f"Total rows: {len(df)}")
            st.write(f"Alert rows: {len(alerts_df)}")

            st.download_button(
                "Download cleaned CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="behavior_monitoring_output.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download alerts CSV",
                data=alerts_df.to_csv(index=False).encode("utf-8"),
                file_name="behavior_alerts_only.csv",
                mime="text/csv"
            )                   