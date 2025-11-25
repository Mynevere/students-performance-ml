import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import subprocess

DATA_PATH = "StudentsPerformance.csv"
OUTPUT_DIR = "outputs"

def load_and_preprocess():
    data = pd.read_csv(DATA_PATH)
    data = pd.get_dummies(data, drop_first=True)

    bins = [0, 50, 70, 100]
    labels = [0, 1, 2]
    data["performance_level"] = pd.cut(data["math score"], bins=bins, labels=labels, include_lowest=True)

    X = data.drop(["math score", "performance_level"], axis=1)
    y = data["performance_level"]
    return X, y

def train_and_evaluate():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    os.makedirs("models/SVM", exist_ok=True)
    joblib.dump(model, "models/SVM/student_performance_svm_model.pkl")
    joblib.dump(scaler, "models/SVM/svm_model_scaler.pkl")
    joblib.dump(X.columns, "models/SVM/svm_model_columns.pkl")

    os.makedirs(f"{OUTPUT_DIR}/metrics", exist_ok=True)
    with open(f"{OUTPUT_DIR}/metrics/svm_metrics.txt", "w") as f:
        f.write(f"Accuracy={accuracy:.4f}\nPrecision={precision:.4f}\nRecall={recall:.4f}\nF1={f1:.4f}\n")

    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SVM - Confusion Matrix")
    plt.savefig(f"{OUTPUT_DIR}/figures/svm_confusion_matrix.png")
    plt.close()

    # PCA Decision Boundaries
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_test)
    model_pca = SVC(kernel="rbf", probability=True, random_state=42)
    model_pca.fit(X_vis, y_test)

    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_test, edgecolors="k", cmap="coolwarm", alpha=0.8)
    plt.title("SVM - Decision Boundaries (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig(f"{OUTPUT_DIR}/figures/svm_decision_boundaries.png")
    plt.close()

    # Classification Report Plot
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose().iloc[:3, :3]  # only precision, recall, f1
    metrics_df.plot(kind="bar", figsize=(8,6))
    plt.title("SVM - Classification Report")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/svm_classification_report.png")
    plt.close()

    try:
        subprocess.run(["python", "analysis/generate_comparison.py"], check=True)
        subprocess.run(["python", "analysis/generate_correlation.py"], check=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not run analysis scripts: {e}")

if __name__ == "__main__":
    train_and_evaluate()
