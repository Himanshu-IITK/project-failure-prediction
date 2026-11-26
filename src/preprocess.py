import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from features import engineer_features, IQROutlierCapper


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for imputation, outlier capping,
    categorical one-hot encoding, and feature scaling.
    """
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_capper', IQROutlierCapper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        verbose_feature_names_out=False
    )
    return preprocessor


def run_preprocessing(
    raw_path: str = "data/raw/project_management_data.csv",
    processed_dir: str = "data/processed",
    model_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load raw data and engineer row-wise PM features
    df = pd.read_csv(raw_path)
    df = engineer_features(df)

    # Drop identifier column and separate features (X) from target (y)
    X = df.drop(columns=['project_id', 'project_failed'])
    y = df['project_failed']

    # 2. Stratified Train-Test Split (80/20) BEFORE any fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

    # 3. Fit preprocessor strictly on X_train, then transform both splits
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # 4. Apply SMOTE strictly on the training set
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

    # Convert back to DataFrames for clean storage and inspection
    train_df = pd.DataFrame(X_train_balanced, columns=feature_names)
    train_df['project_failed'] = y_train_balanced.values

    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df['project_failed'] = y_test.values

    # 5. Save processed datasets and fitted preprocessor
    train_df.to_csv(os.path.join(processed_dir, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_processed.csv"), index=False)
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))

    # Verification Summary
    print("=== Preprocessing Summary ===")
    print(f"Total Engineered Features (after One-Hot Encoding): {len(feature_names)}")
    print(f"Train Set BEFORE SMOTE: {X_train.shape[0]} rows | Failure rate: {y_train.mean():.3f}")
    print(f"Train Set AFTER SMOTE : {train_df.shape[0]} rows | Failure rate: {train_df['project_failed'].mean():.3f}")
    print(f"Test Set (Untouched)  : {test_df.shape[0]} rows  | Failure rate: {test_df['project_failed'].mean():.3f}")
    print(f"Missing Values Remaining -> Train: {train_df.isnull().sum().sum()} | Test: {test_df.isnull().sum().sum()}")
    print(f"\nSaved processed CSVs to '{processed_dir}/' and pipeline to '{model_dir}/preprocessor.joblib'")


if __name__ == "__main__":
    run_preprocessing()