from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints key metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.2f}")
    print(f"Recall: {recall:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
