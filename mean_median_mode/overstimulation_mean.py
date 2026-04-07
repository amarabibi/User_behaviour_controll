import pandas as pd

data = pd.read_csv("overstimulation_dataset.csv")
data.columns = data.columns.str.strip()

numeric_cols = [
    "Age",
    "Sleep_Hours",
    "Screen_Time",
    "Stress_Level",
    "Noise_Exposure",
    "Social_Interaction",
    "Work_Hours",
    "Exercise_Hours",
    "Caffeine_Intake",
    "Multitasking_Habit",
    "Anxiety_Score",
    "Depression_Score",
    "Sensory_Sensitivity",
    "Meditation_Habit",
    "Overthinking_Score",
    "Irritability_Score",
    "Headache_Frequency",
    "Sleep_Quality",
    "Tech_Usage_Hours",
    "Overstimulated"
]

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

print("NUMERIC COLUMNS")
for col in numeric_cols:
    if col in data.columns:
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
            print("No valid data found")
    else:
        print(f"\n{col} not found in CSV")