from src.data_loader import load_data
from src.preprocess import clean_data, feature_engineering, balance_classes
from src.train import train_model
from src.evaluate import evaluate_model
from src.models import get_models
from sklearn.model_selection import train_test_split

# Load dataset
df = load_data("data/NEOWS Dataset.csv")

# Preprocess data
df = clean_data(df)
df = feature_engineering(df)
df = balance_classes(df)

# Save processed data
df.to_csv("data/processed_data.csv", index=False)

# Split dataset
X = df.drop(columns=['Hazardous'])
y = df['Hazardous']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = get_models()
trained_models = {name: train_model(model, X_train, y_train) for name, model in models.items()}

# Evaluate models
for name, model in trained_models.items():
    print(f"Evaluating {name} model:")
    evaluate_model(model, X_test, y_test)
    print("\n" + "-"*50 + "\n")
