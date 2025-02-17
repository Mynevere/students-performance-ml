import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    try:
        data = pd.read_csv(filepath)
        # Handle missing values
        data = data.dropna()
        # Encode categorical variables
        data = pd.get_dummies(data, drop_first=True)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_features_and_target(data):
    """Prepare features and target variables."""
    X = data.drop(['math score'], axis=1)
    # Categorize math scores into Low, Medium, and High
    y = pd.cut(data['math score'], 
               bins=[0, 50, 70, 100], 
               labels=['Low', 'Medium', 'High'])
    y = y.astype(str)
    
    # Remove any NaN values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    return X, y

def train_model(X_train, y_train):
    """Train the SVM model."""
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = SVC(kernel='linear', probability=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model
    model.fit(X_train_scaled, y_train)
    return model, scaler

def save_artifacts(model, scaler, columns, model_path, columns_path, scaler_path):
    """Save model and related artifacts."""
    artifacts = [
        (model, model_path),
        (columns.tolist(), columns_path),  # Convert Index to list
        (scaler, scaler_path)
    ]
    
    for artifact, path in artifacts:
        try:
            with open(path, "wb") as f:
                pickle.dump(artifact, f)
        except Exception as e:
            logger.error(f"Error saving to {path}: {str(e)}")
            raise

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate the model and save results."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # Save and print the report
    try:
        with open("svm_model_evaluation.txt", "w") as file:
            file.write(report)
        logger.info("Classification Report:\n%s", report)
    except Exception as e:
        logger.error(f"Error saving evaluation report: {str(e)}")
        raise

def main():
    try:
        # Load and preprocess data
        data = load_and_preprocess_data("../DataSet/StudentsPerformance.csv")
        
        # Prepare features and target
        X, y = prepare_features_and_target(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model, scaler = train_model(X_train, y_train)
        
        # Save artifacts
        save_artifacts(
            model=model,
            scaler=scaler,
            columns=X.columns,
            model_path="student_performance_svm_model.pkl",
            columns_path="svm_model_columns.pkl",
            scaler_path="svm_model_scaler.pkl"
        )
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, scaler)
        
        logger.info("Model training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
