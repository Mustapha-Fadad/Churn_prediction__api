# Customer Churn Prediction

Predict telecom customer churn with scikit-learn pipelines, Random Forest, and XGBoost. A FastAPI service exposes the trained XGBoost model for individual customer predictions.

The project includes data cleaning, feature engineering, model training, evaluation, and an exploratory analysis notebook.

## Project structure

```text
churn_data/                 Telco customer churn CSV
notebooks/eda.ipynb         Exploratory data analysis
src/
  data/load_data.py        Data loading and cleaning
  features/               Feature engineering transformer
  models/                 Training, evaluation, and explanation scripts
  artifacts/              Saved pipelines, test data, and threshold metadata
  api/app.py              FastAPI application
requirements.txt          Pinned Python dependencies
Dockerfile                API container definition
```

## Setup

Use Python 3.11 or newer. Run all commands from the project root.

```sh
python -m venv venv
```

Activate the environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source venv/bin/activate
```

Install dependencies:

```sh
python -m pip install -r requirements.txt
```

## Dataset and features

Training expects the dataset at:

```text
churn_data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The CSV contains customer demographics, subscriptions, billing information, and a `Churn` target with `Yes`/`No` values. The loader removes rows with missing values, converts `TotalCharges` to numeric, encodes the target, and drops `customerID`.

The feature transformer adds average revenue per month of tenure (`ARPU`), tenure groups, and the ratio of monthly charges to ARPU. Each model pipeline includes feature engineering and preprocessing.

## Train and evaluate

Run the scripts in this order to generate the models, shared test set, and API threshold metadata:

```sh
python -m src.models.train_random_forest
python -m src.models.train_xgb
python -m src.models.evaluate_rf
python -m src.models.evaluate_xgb
```

Both training scripts use a stratified 80/20 split with `random_state=42`. Random Forest training saves `rf_pipeline.pkl` and `test_data.csv`. XGBoost training performs a randomized hyperparameter search with three-fold cross-validation and saves `xgb_pipeline.pkl`.

Evaluation prints ROC-AUC, PR-AUC, a confusion matrix, and a classification report, and displays ROC and precision-recall plots. XGBoost evaluation also saves the threshold that maximizes F1 on the test set to `xgb_metadata.json`. Because the threshold is selected on the test set, these results are not an independent estimate of performance after threshold tuning.

All generated files are stored in `src/artifacts/`. Retraining overwrites the corresponding artifacts; rerun XGBoost evaluation after retraining to refresh the API threshold.

## Run the API

The API needs `src/artifacts/xgb_pipeline.pkl` and `src/artifacts/xgb_metadata.json` at startup. Use the included artifacts or generate them with the commands above.

```sh
python -m uvicorn src.api.app:app --reload
```

- Health check: <http://127.0.0.1:8000/>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Prediction endpoint: `POST /predict`

Submit this JSON using the interactive documentation or an HTTP client:

```json
{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 850.2,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "PaperlessBilling": "Yes"
}
```

The response contains `churn_probability`, `prediction`, and `threshold_used`. A prediction of `1` indicates churn; `0` indicates no churn. The API predicts churn when the probability exceeds the threshold saved in the metadata.

## Docker

With the XGBoost pipeline and metadata present in `src/artifacts/`:

```sh
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
```

Open <http://localhost:8000/docs> to try the API.

## Development notes

- `notebooks/eda.ipynb` contains exploratory analysis. Install Jupyter separately if needed; it is not included in `requirements.txt`.
- `src/models/explain_xgb.py` contains SHAP analysis code, but its model path and preprocessing need updating before it can run with the current pipeline layout.
- `.gitignore` excludes local environments, environment files, caches, editor settings, and generated artifacts from new Git additions. Files already tracked by Git remain tracked.
