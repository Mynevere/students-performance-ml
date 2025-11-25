import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
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

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    os.makedirs("models/RandomForest", exist_ok=True)
    joblib.dump(model, "models/RandomForest/student_performance_randomforest_model.pkl")
    joblib.dump(X.columns, "models/RandomForest/randomforest_model_columns.pkl")

    os.makedirs(f"{OUTPUT_DIR}/metrics", exist_ok=True)
    with open(f"{OUTPUT_DIR}/metrics/randomforest_metrics.txt", "w") as f:
        f.write(f"Accuracy={accuracy:.4f}\nPrecision={precision:.4f}\nRecall={recall:.4f}\nF1={f1:.4f}\n")

    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest - Confusion Matrix")
    plt.savefig(f"{OUTPUT_DIR}/figures/randomforest_confusion_matrix.png")
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances[indices], y=X.columns[indices], hue=X.columns[indices], dodge=False, palette="viridis", legend=False)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Random Forest - Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/randomforest_feature_importance.png")
    plt.close()

    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    y_pred_prob = model.predict_proba(X_test)

    plt.figure(figsize=(7,6))
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{OUTPUT_DIR}/figures/randomforest_roc_curve.png")
    plt.close()
    try:
        subprocess.run(["python", "analysis/generate_comparison.py"], check=True)
        subprocess.run(["python", "analysis/generate_correlation.py"], check=True)
    except Exception as e:
        print(f"Warning: Could not run analysis scripts: {e}")

if __name__ == "__main__":
    train_and_evaluate()
