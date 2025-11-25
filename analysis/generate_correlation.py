import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "StudentsPerformance.csv"
OUTPUT_DIR = "outputs"

def generate_correlation():
    data = pd.read_csv(DATA_PATH)

    # Encode categorical variables for correlation
    data_encoded = pd.get_dummies(data, drop_first=True)

    corr = data_encoded.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap of Dataset Features")
    plt.tight_layout()

    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/figures/correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    generate_correlation()
