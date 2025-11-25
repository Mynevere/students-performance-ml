import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import subprocess

DATA_PATH = "StudentsPerformance.csv"
OUTPUT_DIR = "outputs"

def load_and_preprocess():
    data = pd.read_csv(DATA_PATH)
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop("math score", axis=1)
    y = data["math score"]
    return X, y

def train_and_evaluate():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")

    os.makedirs("models/LinearRegression", exist_ok=True)
    joblib.dump(model, "models/LinearRegression/student_performance_linearregression_model.pkl")
    joblib.dump(X.columns, "models/LinearRegression/linearregression_model_columns.pkl")

    os.makedirs(f"{OUTPUT_DIR}/metrics", exist_ok=True)
    with open(f"{OUTPUT_DIR}/metrics/linearregression_metrics.txt", "w") as f:
        f.write(f"MAE={mae:.4f}\nMSE={mse:.4f}\nRMSE={rmse:.4f}\nR2={r2:.4f}\n")

    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)


    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Linear Regression - Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/linearregression_scatter.png")
    plt.close()

    # Residuals Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, bins=20, kde=True, color="green")
    plt.title("Linear Regression - Residuals Distribution")
    plt.xlabel("Residuals")
    plt.savefig(f"{OUTPUT_DIR}/figures/linearregression_residuals.png")
    plt.close()

    # Residuals vs Predicted
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.6, color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Linear Regression - Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/linearregression_residuals_vs_predicted.png")
    plt.close()

    try:
        subprocess.run(["python", "analysis/generate_comparison.py"], check=True)
        subprocess.run(["python", "analysis/generate_correlation.py"], check=True)
    except Exception as e:
        print(f"Warning: Could not run analysis scripts: {e}")

if __name__ == "__main__":
    train_and_evaluate()
