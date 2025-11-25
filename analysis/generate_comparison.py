import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "outputs"

def load_metrics(file_path, model_name):
    metrics = {}
    with open(file_path, "r") as f:
        for line in f:
            k, v = line.strip().split("=")
            metrics[k] = float(v)
    metrics["Model"] = model_name
    return metrics

def generate_comparison():
    metrics_files = {
        "Linear Regression": f"{OUTPUT_DIR}/metrics/linearregression_metrics.txt",
        "Random Forest": f"{OUTPUT_DIR}/metrics/randomforest_metrics.txt",
        "SVM": f"{OUTPUT_DIR}/metrics/svm_metrics.txt",
    }

    data = []
    for model, file_path in metrics_files.items():
        if os.path.exists(file_path):
            data.append(load_metrics(file_path, model))
        else:
            print(f"⚠️ Metrics file not found for {model}: {file_path}")

    df = pd.DataFrame(data)

    # Melt for plotting
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted, palette="Set2")
    plt.title("Comparison of Models - Accuracy, Precision, Recall, F1")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.legend(title="Model")
    plt.tight_layout()

    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/figures/comparison_metrics.png")
    plt.close()

if __name__ == "__main__":
    generate_comparison()
