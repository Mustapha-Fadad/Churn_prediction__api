import pandas as pd


def load_clean_data(path: str) -> pd.DataFrame:
    """
    Load raw churn dataset and perform basic cleaning.
    """

    df = pd.read_csv(path)
    
    # Drop rows with missing values
    df= df.dropna()

    # Fix TotalCharges type
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0

    # Encode target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Drop ID column
    df = df.drop('customerID', axis=1)

    return df
