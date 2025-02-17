import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully with {data.shape[0]} rows")
        
        # Handle missing values
        initial_rows = len(data)
        data = data.dropna()
        if len(data) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(data)} rows with missing values")
        
        # Encode categorical variables
        data = pd.get_dummies(data, drop_first=True)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_model(X, y):
    """Train the linear regression model with cross-validation."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Average CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using multiple metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    return metrics

def save_model_artifacts(model, columns, metrics, model_path, columns_path, eval_path):
    """Save model, columns, and evaluation results."""
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        with open(columns_path, "wb") as f:
            pickle.dump(columns, f)
            
        with open(eval_path, "w") as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value}\n")
        
        logger.info("Model artifacts saved successfully")
    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise

def main():
    try:
        # Load and preprocess data
        data = load_and_preprocess_data("../DataSet/StudentsPerformance.csv")
        
        # Split into features and target
        X = data.drop(['math score'], axis=1)
        y = data['math score']
        
        # Train model
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save all artifacts
        save_model_artifacts(
            model=model,
            columns=X.columns,
            metrics=metrics,
            model_path="student_performance_linearregression_model.pkl",
            columns_path="linearregression_model_columns.pkl",
            eval_path="linearregression_model_evaluation.txt"
        )
        
        # Log results
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
