import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir='outputs/eda'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("Shape:\n" + str(df.shape) + "\n\n")
        f.write("Missing Values:\n" + str(df.isnull().sum()) + "\n\n")
        f.write("Duplicates:\n" + str(df.duplicated().sum()) + "\n")

    # 2. Visualizations
    # Target Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='is_canceled', data=df)
    plt.title("Cancellation Distribution")
    plt.savefig(f"{output_dir}/target_distribution.png")
    plt.close()

    # Correlation Matrix (Numeric only)
    plt.figure(figsize=(10,8))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    print(f"EDA Report saved to {output_dir}")