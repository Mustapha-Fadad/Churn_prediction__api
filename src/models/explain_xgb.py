import sys
from pathlib import Path
import joblib
import shap
import pandas as pd

# Project root
root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path / "src"))

from data.load_data import load_clean_data
from features.feature_engineering import FeatureEngineer


def explain_model(data_path: str):

    # Load pipeline
    pipeline = joblib.load("models/xgb_pipeline.pkl")

    # Load and process data
    df = load_clean_data(data_path)
    

    X = df.drop("Churn", axis=1)

    # Transform data using pipeline preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)


    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Plot summary
    shap.summary_plot(shap_values, X_transformed)


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[2]
    data_path = base_path / "churn_data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    explain_model(str(data_path))