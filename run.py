from src.preprocess import preprocess_data
from src.eda import perform_eda
from src.features import feature_engineering
from src.pycaret_train import train_with_pycaret

# Define file paths
input_file = "data/NEOWS Dataset.csv"
cleaned_file = "data/NEOWS_Cleaned.csv"
processed_file = "data/NEOWS_Processed_SMOTE.csv"  # New file after SMOTE
eda_output_dir = "reports/eda/"
model_output_dir = "models/"

# Step 1: Run Preprocessing
preprocess_data(input_file, cleaned_file)

# Step 2: Perform EDA (Saves plots)
perform_eda(cleaned_file, eda_output_dir)

# Step 3: Feature Engineering with SMOTE
feature_engineering(cleaned_file, processed_file)

# Step 4: Train & Compare Models Using PyCaret
train_with_pycaret(processed_file, model_output_dir)
