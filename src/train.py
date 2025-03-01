from sklearn.model_selection import train_test_split
from models import get_models
import pandas as pd

def train_model(model, X_train, y_train):
    """Trains a model and returns it."""
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    df = pd.read_csv("../data/processed_data.csv")
    X = df.drop(columns=['Hazardous'])
    y = df['Hazardous']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = get_models()
    
    trained_models = {name: train_model(model, X_train, y_train) for name, model in models.items()}
