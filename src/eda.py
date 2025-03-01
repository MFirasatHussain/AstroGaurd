import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(input_file, output_dir="reports/eda/"):
    """
    Loads the cleaned dataset, generates key visualizations, and saves them as images.
    """
    # Load dataset
    df = pd.read_csv(input_file)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Display basic statistics
    print("\n🔹 Basic Statistics:")
    print(df.describe())

    # Save statistics as a text file
    stats_file = os.path.join(output_dir, "basic_statistics.txt")
    with open(stats_file, "w") as f:
        f.write(df.describe().to_string())
    print(f"✅ Basic statistics saved to {stats_file}")

    # Plot and save class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=df["Hazardous"], palette="viridis")
    plt.title("Distribution of Hazardous vs Non-Hazardous Asteroids")
    plt.xlabel("0 = Not Hazardous | 1 = Hazardous")
    plt.ylabel("Count")
    class_dist_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(class_dist_path)
    print(f"✅ Class distribution plot saved to {class_dist_path}")
    plt.close()

    # 🔹 FIX: Remove non-numeric columns before computing correlations
    df_numeric = df.select_dtypes(include=["number"])  # Keep only numeric columns

    # Plot and save correlation heatmap
    plt.figure(figsize=(12,6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    print(f"✅ Correlation heatmap saved to {heatmap_path}")
    plt.close()

    print("✅ EDA completed. All visualizations saved in:", output_dir)
