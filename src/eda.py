import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_heatmap(df):
    """Generates a correlation heatmap."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()

def class_distribution(df):
    """Shows class balance in dataset."""
    df['Hazardous'].value_counts().plot(kind='bar')
    plt.title("Class Distribution")
    plt.show()
