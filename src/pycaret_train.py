import pandas as pd
from pycaret.classification import *

def train_with_pycaret(input_file, model_output_dir="models/", report_output="reports/model_performance.csv"):
    """
    Uses PyCaret to train multiple models, compare them, and save the best one.
    Also stores model performance metrics in a CSV file.
    """
    # Load dataset
    df = pd.read_csv(input_file)

    # Ensure target variable is categorical (0 or 1)
    df["Hazardous"] = df["Hazardous"].astype(int)

    # Setup PyCaret
    clf_setup = setup(data=df, target="Hazardous", session_id=42, normalize=True)

    # Compare models and get results as a DataFrame
    model_comparison = compare_models(n_select=5, sort="Accuracy")
    model_results = pull()  # Extract performance results table

    # Save results locally
    model_results.to_csv(report_output, index=False)
    print(f"✅ Model performance saved to {report_output}")

    # Save the best model
    best_model = model_comparison
    save_model(best_model, f"{model_output_dir}/best_pycaret_model")

    print(f"✅ Best model trained & saved at {model_output_dir}/best_pycaret_model.pkl")

    return best_model
