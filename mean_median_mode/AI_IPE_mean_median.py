import pandas as pd

data = pd.read_csv("AI_IPE_IoT_BigData_Dataset.csv")

numeric_cols = [
    'age', 'heart_rate', 'eye_gaze_focus_level', 'head_nod_count',
    'body_posture_score', 'speech_frequency', 'environment_noise_level',
    'duration_minutes', 'quiz_score', 'engagement_level'
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

for col in numeric_cols:
    print(f"\n{col}")
    print("Mean:", data[col].mean())
    print("Median:", data[col].median())
    mode_values = data[col].mode()
    if not mode_values.empty:
        print("Mode:", mode_values.iloc[0])
    else:
        print("Mode: No mode found")