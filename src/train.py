import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

def train_models(input_file, model_output_dir="models/"):
    """
    Trains Decision Tree, Random Forest, and AdaBoost models, then saves them.
    """
    # Load dataset
    df = pd.read_csv(input_file)

    # Ensure target variable is categorical (0 or 1)
    df["Hazardous"] = df["Hazardous"].astype(int)

    # Split into features & target
    X = df.drop(columns=['Hazardous'])
    y = df['Hazardous']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(n_estimators=50)
    }

    # Ensure model directory exists
    import os
    os.makedirs(model_output_dir, exist_ok=True)

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  # ✅ Fix: y_train is now properly formatted as categorical (0/1)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        # Save model
        model_path = f"{model_output_dir}/{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"✅ {name} trained & saved at {model_path} | Accuracy: {acc:.2f}")

    return results
