import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

OUTPUT_FOLDER = Path("output")

def main():
    # 1. Load the pre-split datasets
    train_path = OUTPUT_FOLDER / "train_ml_ready.csv"
    test_path = OUTPUT_FOLDER / "test_ml_ready.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("Error: Split data files not found. Run your feature engineering script first.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Define your target variable / prediction label
    # Setting 'user_behavior_class' as default. Update if you are predicting another column.
    target_col = "user_behavior_class"
    
    if target_col not in train_df.columns:
        print(f"Error: Target column '{target_col}' not found in the dataset.")
        return

    # 3. Separate features (X) and target labels (y)
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 4. Automatically categorize features into Numeric and Categorical text
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=[object, "category", bool]).columns.tolist()

    print(f"Found {len(numeric_features)} Numeric Features.")
    print(f"Found {len(categorical_features)} Categorical/Boolean Features: {categorical_features}")

    # 5. Build isolated sub-pipelines for processing both types of data
    # Numerical pipeline: Fills NaNs with the median value
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Categorical pipeline: Fills NaNs, converts mixed booleans to strings, then applies One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value')),
        # FIXED: Converts mixed columns (strings, booleans) to standard string representations
        ('force_string', FunctionTransformer(lambda x: x.astype(str), validate=False)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 6. Combine individual pipelines into a master dataset processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 7. Fit on training data and transform both sets
    print("\nTransforming data columns...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 8. Retrieve updated column names for evaluation clarity
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
    final_feature_names = numeric_features + encoded_cat_cols

    # Convert processed matrices back into structured DataFrames
    X_train_final = pd.DataFrame(X_train_processed, columns=final_feature_names)
    X_test_final = pd.DataFrame(X_test_processed, columns=final_feature_names)

    print("\n--- Processing Completed Successfully ---")
    print(f"Final Train Matrix Shape: {X_train_final.shape}")
    print(f"Final Test Matrix Shape:  {X_test_final.shape}")
    
    return X_train_final, X_test_final, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main()
