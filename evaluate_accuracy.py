import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUTPUT_FOLDER = Path("output")

# FIX: Added this custom function definition so joblib can successfully unpickle the pipeline
def force_string_conversion(x):
    return x.astype(str)

def main():
    # 1. Load the production-ready model pipeline and label encoders
    print("Loading model pipeline and label mappings...")
    pipeline = joblib.load(OUTPUT_FOLDER / "behavior_monitoring_pipeline.pkl")
    label_encoder = joblib.load(OUTPUT_FOLDER / "target_labels.pkl")
    
    # 2. Load the unseen test dataset
    test_path = OUTPUT_FOLDER / "test_ml_ready.csv"
    if not test_path.exists():
        print(f"Error: Test file not found at {test_path}")
        return
        
    test_df = pd.read_csv(test_path, low_memory=False)
    
    # 3. Separate your target column from the features
    target_col = "user_behavior_class"
    
    # Remove any missing target labels from the test evaluation data
    test_df = test_df.dropna(subset=[target_col])
    
    X_test = test_df.drop(columns=[target_col])
    y_test_raw = test_df[target_col]
    
    # 4. Map the true text labels into the exact numbers the model expects
    y_test_true = label_encoder.transform(y_test_raw.astype(str))
    
    # 5. Generate predictions using the full pipeline 
    print(f"Generating behavior predictions for {len(X_test)} records...")
    y_pred = pipeline.predict(X_test)
    
    # 6. Calculate and display core performance metrics
    accuracy = accuracy_score(y_test_true, y_pred)
    
    print("\n=========================================")
    print("      MODEL ACCURACY TEST REPORT         ")
    print("=========================================")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("-----------------------------------------")
    
    print("Detailed Classification Performance:")
    print(classification_report(
        y_test_true, 
        y_pred, 
        target_names=label_encoder.classes_
    ))
    
    print("-----------------------------------------")
    print("Confusion Matrix Matrix (Raw Count):")
    cm = confusion_matrix(y_test_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=[f"Pred_{c}" for c in label_encoder.classes_])
    print(cm_df)
    print("=========================================")

if __name__ == "__main__":
    main()
