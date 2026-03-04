import sys
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Fix module path
root_path = Path(__file__).resolve().parents[2]


from src.data.load_data import load_clean_data
from src.features.feature_engineering import FeatureEngineer

def train_model(data_path: str):

    # Load data
    df = load_clean_data(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Identify feature types
    numeric_features = list(X.select_dtypes(include=["int64", "float64"]).columns)
    numeric_features += ["ARPU", "price_ratio"]

    categorical_features = list(X.select_dtypes(include=["object"]).columns)
    categorical_features += ["tenure_group"]

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    # Save pipeline
    model_path = base_path / "src" / "artifacts"
    model_path.mkdir(exist_ok=True)

    joblib.dump(pipeline, model_path / "rf_pipeline.pkl")

    # Save test set
    test_data = X_test.copy()
    test_data["Churn"] = y_test
    test_data.to_csv(model_path / "test_data.csv", index=False)

    print("Pipeline and test set saved successfully.")


if __name__ == "__main__":

    base_path = Path(__file__).resolve().parents[2]
    data_path = base_path / "churn_data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    train_model(str(data_path))
