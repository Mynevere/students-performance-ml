import pandas as pd
import pickle
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Model configuration
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'score_bins': [0, 50, 70, 100],
    'score_labels': ['Low', 'Medium', 'High']
}

def load_and_preprocess_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the dataset.
    """
    try:
        data = pd.read_csv(filepath)
        data = data.dropna()
        
        # Encode categorical variables
        data = pd.get_dummies(data, drop_first=True)
        
        # Split into features and target
        X = data.drop(['math score'], axis=1)
        y = pd.cut(data['math score'], 
                  bins=CONFIG['score_bins'], 
                  labels=CONFIG['score_labels'])
        
        # Clean and validate target
        y = y.astype(str)
        mask = ~y.isna()
        return X[mask], y[mask]
    
    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train the Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=CONFIG['n_estimators'],
        random_state=CONFIG['random_state'],
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def save_artifacts(model: RandomForestClassifier, columns: pd.Index, report: str) -> None:
    """
    Save model artifacts to disk.
    """
    try:
        with open("student_performance_randomforestmodel.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open("randomforestmodel_columns.pkl", "wb") as f:
            pickle.dump(columns, f)
            
        with open("randomforestmodel_evaluation.txt", "w") as f:
            f.write(report)
            
        logging.info("Model artifacts saved successfully")
    except Exception as e:
        logging.error(f"Error saving artifacts: {str(e)}")
        raise

def plot_confusion_matrix(y_test: pd.Series, y_pred: pd.Series) -> None:
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        X, y = load_and_preprocess_data("../DataSet/StudentsPerformance.csv")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state']
        )
        
        # Train model
        logging.info("Training model...")
        model = train_model(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        logging.info(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred)
        
        # Save artifacts
        save_artifacts(model, X.columns, report)
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
