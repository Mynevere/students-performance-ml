import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score


DATA_PATH = "StudentsPerformance.csv"
OUTPUT_DIR = "outputs/figures"

data = pd.read_csv(DATA_PATH)
data = pd.get_dummies(data, drop_first=True)

# For regression (predict math score)
X_reg = data.drop("math score", axis=1)
y_reg = data["math score"]

# For classification (categorical target)
bins = [0, 50, 70, 100]
labels = [0, 1, 2]  # 0=Low, 1=Average, 2=High
data["performance_level"] = pd.cut(y_reg, bins=bins, labels=labels, include_lowest=True)
X_clf = data.drop(["math score", "performance_level"], axis=1)
y_clf = data["performance_level"]


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42)
}


splits = [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]
results = {model: [] for model in models.keys()}

for train_size, test_size in splits:
    # Linear Regression
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, train_size=train_size, test_size=test_size, random_state=42
    )
    lr = models["Linear Regression"]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results["Linear Regression"].append(r2_score(y_test, y_pred))

    # Random Forest
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, train_size=train_size, test_size=test_size, random_state=42, stratify=y_clf
    )
    rf = models["Random Forest"]
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["Random Forest"].append(accuracy_score(y_test, y_pred))

    # SVM (scaled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, train_size=train_size, test_size=test_size, random_state=42, stratify=y_clf
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm = models["SVM (RBF)"]
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    results["SVM (RBF)"].append(accuracy_score(y_test, y_pred))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {
    "Linear Regression": cross_val_score(models["Linear Regression"], X_reg, y_reg, cv=kfold, scoring="r2").mean(),
    "Random Forest": cross_val_score(models["Random Forest"], X_clf, y_clf, cv=kfold, scoring="accuracy").mean(),
    "SVM (RBF)": cross_val_score(
        Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", random_state=42))]),
        X_clf, y_clf, cv=kfold, scoring="accuracy"
    ).mean()
}


df_results = pd.DataFrame(results, index=["70/30", "80/20", "90/10"])
df_results.loc["CrossVal"] = cv_results

print("=== Model Stability Results ===")
print(df_results.round(3))


os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(["70/30", "80/20", "90/10"], df_results["Linear Regression"][:-1], marker="^", label="Linear Regression (R²)")
plt.plot(["70/30", "80/20", "90/10"], df_results["Random Forest"][:-1], marker="o", label="Random Forest (Accuracy)")
plt.plot(["70/30", "80/20", "90/10"], df_results["SVM (RBF)"][:-1], marker="s", label="SVM (Accuracy)")

plt.title("Model Stability Across Train/Test Splits", fontsize=12, weight="bold")
plt.xlabel("Train/Test Split")
plt.ylabel("Performance")
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_stability.png"), dpi=300)
plt.close()

print("Figure 'model_stability.png' saved successfully in outputs/figures/")
