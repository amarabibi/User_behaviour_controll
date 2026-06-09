import pandas as pd
import joblib
from pathlib import Path

OUTPUT_FOLDER = Path("output")

# 1. Load the pre-trained assets into runtime memory
pipeline = joblib.load(OUTPUT_FOLDER / "behavior_monitoring_pipeline.pkl")
label_encoder = joblib.load(OUTPUT_FOLDER / "target_labels.pkl")

# 2. Simulate raw, fresh behavioral data arriving from a user (e.g., inside an API or app layer)
# Pass a single dictionary or dataframe containing the columns your model trained on
new_telemetry_data = pd.DataFrame([{
    "gender": "male",
    "major": "computer_science",
    "device_used": "laptop",
    "age": 22,
    "login_frequency_30_days": 15,
    # Add whatever features are active in your feature sets...
}])

# 3. Predict classification labels directly from raw un-encoded shapes!
numeric_prediction_index = pipeline.predict(new_telemetry_data)[0]

# 4. Decode the numeric index back to original string classifications
readable_behavior_class = label_encoder.inverse_transform([numeric_prediction_index])[0]

print(f"\nLive User Classification Result: {readable_behavior_class}")
