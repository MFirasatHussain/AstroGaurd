import pandas as pd

def preprocess_data(input_file, output_file):
    """
    Loads the dataset, cleans unnecessary columns, converts categorical features,
    and saves the cleaned dataset.
    """
    # Load dataset
    df = pd.read_csv(input_file)

    # Define columns to drop (redundant measurements & non-essential details)
    columns_to_drop = [
        'Neo Reference ID', 'Name', 'Close Approach Date', 'Epoch Date Close Approach',
        'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
        'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Relative Velocity km per hr', 'Miles per hour',
        'Miss Dist.(lunar)', 'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Orbit ID',
        'Orbit Determination Date', 'Equinox'
    ]

    # Drop unnecessary columns
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Convert 'Hazardous' column to numeric (1 = Hazardous, 0 = Not Hazardous)
    df['Hazardous'] = df['Hazardous'].astype(int)

    # Save the cleaned dataset
    df.to_csv(output_file, index=False)
    print(f"✅ Preprocessing complete. Cleaned dataset saved to: {output_file}")

    return df  # Returning df for validation
