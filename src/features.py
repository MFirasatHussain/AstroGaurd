import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def feature_engineering(input_file, output_file):
    """
    Encodes categorical variables, applies SMOTE to balance data, scales numerical features, 
    and saves the processed dataset.
    """
    # Load dataset
    df = pd.read_csv(input_file)

    # Encode categorical variables (Orbiting Body)
    if 'Orbiting Body' in df.columns:
        le = LabelEncoder()
        df['Orbiting Body'] = le.fit_transform(df['Orbiting Body'])

    # Separate features and target
    X = df.drop(columns=['Hazardous'])
    y = df['Hazardous']

    # Apply SMOTE to balance the classes
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convert back to DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['Hazardous'] = y_resampled

    # Scale numerical features
    scaler = StandardScaler()
    numeric_cols = df_resampled.select_dtypes(include=["number"]).columns
    df_resampled[numeric_cols] = scaler.fit_transform(df_resampled[numeric_cols])

    # Save processed dataset
    df_resampled.to_csv(output_file, index=False)
    print(f"✅ SMOTE applied & dataset saved to: {output_file}")

    return df_resampled
