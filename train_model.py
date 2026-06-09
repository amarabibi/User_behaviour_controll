import pandas as pd
import numpy as np
from pathlib import Path
import joblib  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

OUTPUT_FOLDER = Path("output")

# Named top-level function so joblib can serialize it without a pickle crash
def force_string_conversion(x):
    return x.astype(str)

def main():
    # 1. Load the pre-split datasets
    train_path = OUTPUT_FOLDER / "train_ml_ready.csv"
    test_path = OUTPUT_FOLDER / "test_ml_ready.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("Error: Split data files not found. Run your feature engineering script first.")
        return

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    # 2. Define your target variable
    target_col = "user_behavior_class"
    
    if target_col not in train_df.columns:
        print(f"Error: Target column '{target_col}' not found in the dataset.")
        return

    train_df = train_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])

    # 3. Separate features (X) and target labels (y)
    X_train_raw = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # FIX: Expanded leaky features list to remove target-overlapping device logs
    # and safely drop 'app_usage_time_z' without any trailing underscore issues.
    leaky_features = [
        'trigger_alert', 'high_stress_hr', 'behavior_length', 
        'stress_level_z', 'heart_rate_z', 'app_usage_time_z',
        'data_usage_mbday', 'app_usage_time_minday', 'app_usage_time', 
        'number_of_apps_installed', 'battery_drain_mahday', 'screen_on_time_hoursday'
    ]
    X_train_raw = X_train_raw.drop(columns=leaky_features, errors='ignore')
    X_test_raw = X_test_raw.drop(columns=leaky_features, errors='ignore')

    # Dynamically drop columns that are 100% empty (All NaNs)
    completely_empty_cols = X_train_raw.columns[X_train_raw.isna().all()].tolist()
    if completely_empty_cols:
        print(f"\nDropping {len(completely_empty_cols)} columns that contain only missing values...")
        X_train = X_train_raw.drop(columns=completely_empty_cols)
        X_test = X_test_raw.drop(columns=completely_empty_cols, errors='ignore')
    else:
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()

    # Encode target labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train.astype(str))
    y_test_encoded = label_encoder.transform(y_test.astype(str))

    # 4. Separate feature types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=[object, "category", "str", "string", bool]).columns.tolist()

    print(f"\nFound {len(numeric_features)} active Numeric Features.")
    print(f"Found {len(categorical_features)} active Categorical/Boolean Features.")

    # 5. Build transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value')),
        ('force_string', FunctionTransformer(force_string_conversion, validate=False)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 6. Combine individual pipelines into a master dataset processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 7. Construct Full Unified Pipeline
    print("\nInitializing complete Scikit-Learn Pipeline...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,        
        class_weight='balanced' 
    )

    full_production_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 8. Train the pipeline
    print("\nTraining production model pipeline...")
    full_production_pipeline.fit(X_train, y_train_encoded)
    print("Model Training Complete.")

    # 9. Evaluate Performance
    y_pred = full_production_pipeline.predict(X_test)
    
    print("\n================ MODEL PERFORMANCE REPORT ================")
    print(f"Overall Accuracy: {accuracy_score(y_test_encoded, y_pred):.4f}\n")
    print("Classification Metrics Breakdown:")
    print(classification_report(
        y_test_encoded, 
        y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # 10. Save assets
    pipeline_save_path = OUTPUT_FOLDER / "behavior_monitoring_pipeline.pkl"
    labels_save_path = OUTPUT_FOLDER / "target_labels.pkl"
    
    joblib.dump(full_production_pipeline, pipeline_save_path)
    joblib.dump(label_encoder, labels_save_path)
    
    print("\n--- Saving Production Assets Complete ---")
    print(f"Saved inference pipeline to: {pipeline_save_path}")
    print(f"Saved target label mappings to: {labels_save_path}")
    
    # Extract and save new feature importance metrics
    final_feature_names = numeric_features + preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
    feature_importance_df = pd.DataFrame({
        'Feature': final_feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv(OUTPUT_FOLDER / "feature_importances.csv", index=False)

if __name__ == "__main__":
    main()
