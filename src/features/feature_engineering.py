import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

            # ARPU
        X['ARPU'] = X['TotalCharges'] / X['tenure']
        X.loc[X['tenure'] == 0, 'ARPU'] = 0

        # Tenure group
        X['tenure_group'] = pd.cut(
          X['tenure'],
          bins=[0, 12, 24, 48, 72],
          labels=['0-1Y', '1-2Y', '2-4Y', '4Y+']
        )

        # Price ratio
        X['price_ratio'] = X['MonthlyCharges'] / (X['ARPU'] + 1e-5)

        return X