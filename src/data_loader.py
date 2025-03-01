import pandas as pd

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def save_data(df, filepath):
    """Save processed dataset to CSV."""
    df.to_csv(filepath, index=False)
