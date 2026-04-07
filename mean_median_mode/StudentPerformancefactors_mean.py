import pandas as pd

data = pd.read_csv("StudentPerformanceFactors.csv")
data.columns = data.columns.str.strip()

numeric_cols = [
    "Hours_Studied",
    "Attendance",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "Peer_Influence",
    "Physical_Activity",
    "Distance_from_Home",
    "Exam_Score"
]

text_cols = [
    "Parental_Involvement",
    "Internet_Access",
    "School_Type",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Gender"
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