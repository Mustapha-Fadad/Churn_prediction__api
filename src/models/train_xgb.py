# src/models/train_xgb.py

import sys
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

# Project root handling
root_path = Path(__file__).resolve().parents[2]


from src.data.load_data import load_clean_data
from src.features.feature_engineering import FeatureEngineer


def train_xgb_model(data_path: str):

    # ===============================
    # Load Raw Data
    # ===============================

    df = load_clean_data(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # ===============================
    # Train/Test Split
    # ===============================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===============================
    # Preprocessing
    # ===============================

    numeric_features = list(X.select_dtypes(include=["int64", "float64"]).columns)
    numeric_features += ["ARPU", "price_ratio"]

    categorical_features = list(X.select_dtypes(include=["object"]).columns)
    categorical_features += ["tenure_group"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # ===============================
    # Handle Class Imbalance
    # ===============================

    scale_pos_weight = (
        len(y_train[y_train == 0]) /
        len(y_train[y_train == 1])
    )

    # ===============================
    # XGBoost Model
    # ===============================

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    # ===============================
    # Full Pipeline
    # ===============================

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", xgb)
        ]
    )

    # ===============================
    # Hyperparameter Search
    # ===============================

    param_grid = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__gamma": [0, 0.1, 0.3],
        "model__min_child_weight": [1, 3, 5]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=30,
        scoring="roc_auc",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest ROC-AUC Score:")
    print(search.best_score_)

    # ===============================
    # Save Full Pipeline
    # ===============================

    artifacts_path = root_path / "src" / "artifacts"
    artifacts_path.mkdir(exist_ok=True)

    model_path = artifacts_path / "xgb_pipeline.pkl"
    joblib.dump(search.best_estimator_, model_path)

    print("\nModel saved successfully at:")
    print(model_path)

    return search


# ===============================
# Script Entry
# ===============================

if __name__ == "__main__":

    base_path = Path(__file__).resolve().parents[2]
    data_path = base_path / "churn_data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    train_xgb_model(str(data_path))