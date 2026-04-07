import pandas as pd

data = pd.read_csv("personality_dataset.csv")
data.columns = data.columns.str.strip()

numeric_cols = [
    "Age",
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency"
]

text_cols = [
    "Personality"
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