import joblib
from pathlib import Path

OUTPUT_FOLDER = Path("output")
labels_path = OUTPUT_FOLDER / "target_labels.pkl"

if not labels_path.exists():
    print(f"Error: Could not find the file at {labels_path}")
else:
    # 1. Load the saved LabelEncoder
    label_encoder = joblib.load(labels_path)
    
    print("=========================================")
    print("      TARGET LABEL MAPPING REPORT        ")
    print("=========================================\n")
    
    # 2. Extract and display the class index mapping
    for numeric_index, text_class in enumerate(label_encoder.classes_):
        print(f"  Model Output Index: [{numeric_index}]  ===>  Real Behavior Label: '{text_class}'")
        
    print("\n=========================================")
