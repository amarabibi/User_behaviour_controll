import pandas as pd
import json

with open("user_activity_data.json", "r", encoding="utf-8") as f:
    data_json = json.load(f)

df = pd.DataFrame(data_json)
df.to_csv("user_activity_dataset.csv", index=False)

print("CSV file created successfully.")