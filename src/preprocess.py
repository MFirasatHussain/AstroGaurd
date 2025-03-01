import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def clean_data(df):
    """Handles missing values, removes unnecessary columns."""
    df = df.drop(columns=['Unnecessary_Column1', 'Unnecessary_Column2'])
    df.dropna(inplace=True)
    return df

def feature_engineering(df):
    """Performs transformations and feature selection."""
    scaler = StandardScaler()
    df[['Feature1', 'Feature2']] = scaler.fit_transform(df[['Feature1', 'Feature2']])
    return df

def balance_classes(df):
    """Uses SMOTE to balance dataset."""
    X = df.drop(columns=['Hazardous'])
    y = df['Hazardous']
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return pd.concat([X_resampled, y_resampled], axis=1)
