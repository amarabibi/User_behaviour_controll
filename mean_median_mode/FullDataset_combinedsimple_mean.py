import pandas as pd

data = pd.read_csv("FullDataset_CombinedSimple.csv")
data.columns = data.columns.str.strip()

numeric_cols = [
    "Age",
    "ThermalSatisfaction",
    "IAQSatisfaction",
    "AcousticSatisfaction",
    "VisualSatisfaction",
    "IEQSatisfaction",
    "CLO_Simple",
    "Thermal_Warm",
    "Thermal_Cold",
    "Thermal_Draft",
    "Thermal_Other",
    "IAQ_Stuffy",
    "IAQ_Odour",
    "IAQ_Dry",
    "IAQ_Humid",
    "IAQ_Other",
    "AC_Classroom",
    "AC_Hallway",
    "AC_Device",
    "AC_Outside",
    "AC_Other",
    "Vis_Blackboard",
    "Vis_Underexposure",
    "Vis_Overexposure",
    "Vis_Glare",
    "Vis_Other",
    "LocationBack",
    "LocationFront",
    "Troom",
    "RH",
    "CO2",
    "VOC",
    "Sound",
    "Lighting",
    "EA",
    "Ttrend",
    "Trm",
    "LocationLeft",
    "LocationRight",
    "LocationMiddle",
    "GroupObs",
    "WindowMajority",
    "WindowsClosed",
    "ActivityMajority",
    "Time_Troom",
    "Time_RH",
    "Time_CO2",
    "Time_EA",
    "Time_Trend",
    "Time_Sound",
    "Time_VOC",
    "Time_Lighting",
    "CLO_Detailed",
    "AC_Ventilation",
    "AF"
]

text_cols = [
    "TimeVote",
    "Subgroup",
    "ID",
    "Moment",
    "Student",
    "Gender",
    "Activity",
    "CaseStudy"
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