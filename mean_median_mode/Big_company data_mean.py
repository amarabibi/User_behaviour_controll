import pandas as pd

data = pd.read_csv("Big_company data1 .csv")

numeric_cols = [
    "Historical Pattern Similarity",
    "Real-Time Behavior Score",
    "Login Frequency (30 days)",
    "Login Attempts"
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

for col in numeric_cols:
    print(f"\n{col}")
    print("Mean:", data[col].mean())
    print("Median:", data[col].median())
    mode_vals = data[col].mode()
    if not mode_vals.empty:
        print("Mode:", mode_vals.iloc[0])
    else:
        print("Mode: No mode found")

text_cols = [
    "Anomaly Type",
    "Mitigation Action",
    "Device Used",
    "Status",
    "User Name",
    "Email"
]

for col in text_cols:
    print(f"\n{col}")
    mode_vals = data[col].mode()
    if not mode_vals.empty:
        print("Mode:", mode_vals.iloc[0])
    else:
        print("Mode: No mode found")