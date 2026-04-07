import pandas as pd

data = pd.read_csv("sstudent_engagement_dataset.csv")
data.columns = data.columns.str.strip()

numeric_cols = [
    "Delta_PSD",
    "Theta_PSD",
    "Alpha_PSD",
    "Beta_PSD",
    "Gamma_PSD",
    "Pupil_Dilation",
    "Blink_Rate",
    "Fixation_Duration",
    "Saccade_Velocity",
    "Difficulty_Level",
    "Engagement_Label"
]

text_cols = [
    "Student_ID",
    "Learning_Content_Type"
]

print("NUMERIC COLUMNS")
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        clean_series = data[col].dropna()

        print(f"\n{col}")
        if not clean_series.empty:
            print("Mean:", clean_series.mean())
            print("Median:", clean_series.median())
            mode_vals = clean_series.mode()
            if not mode_vals.empty:
                print("Mode:", mode_vals.iloc[0])
            else:
                print("Mode: No mode found")
        else:
            print("No valid numeric data found")
    else:
        print(f"\n{col} not found in CSV")

print("\nTEXT COLUMNS")
for col in text_cols:
    if col in data.columns:
        clean_series = data[col].dropna()

        print(f"\n{col}")
        if not clean_series.empty:
            mode_vals = clean_series.mode()
            if not mode_vals.empty:
                print("Mode:", mode_vals.iloc[0])
            else:
                print("Mode: No mode found")
        else:
            print("No valid text data found")
    else:
        print(f"\n{col} not found in CSV")