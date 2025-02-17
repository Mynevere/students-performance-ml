import pandas as pd
import pickle
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load the trained models and column mappings
models = {
    "RandomForest": {
        "model": os.path.join('models', 'RandomForest', 'student_performance_randomforestmodel.pkl'),
        "columns": os.path.join('models', 'RandomForest', 'randomforestmodel_columns.pkl')
    },
    "LinearRegression": {
        "model": os.path.join('models', 'LinearRegression', 'student_performance_linearregression_model.pkl'),
        "columns": os.path.join('models', 'LinearRegression', 'linearregression_model_columns.pkl')
    },
    "SVM": {
        "model": os.path.join('models', 'SVM', 'student_performance_svm_model.pkl'),
        "columns": os.path.join('models', 'SVM', 'svm_model_columns.pkl')
    }
}

# Load all models
loaded_models = {}
loaded_columns = {}

for model_name, paths in models.items():
    with open(paths["model"], "rb") as f:
        loaded_models[model_name] = pickle.load(f)
    with open(paths["columns"], "rb") as f:
        loaded_columns[model_name] = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            model_choice = data.get('models')
            student_data = data.get('inputs')

            print(f"Received data: {student_data}")  # Debugging step

            if model_choice not in loaded_models:
                return jsonify({"error": "Invalid model choice"}), 400

            # Convert input into DataFrame
            df = pd.DataFrame([student_data])

            # One-hot encode the input data to match training data format
            df_encoded = pd.get_dummies(df)

            # Ensure the columns are the same as in training
            missing_cols = set(loaded_columns[model_choice]) - set(df_encoded.columns)
            for col in missing_cols:
                df_encoded[col] = 0

            # Reorder columns to match training columns order
            df_encoded = df_encoded[loaded_columns[model_choice]]

            # Make prediction using selected model
            prediction = loaded_models[model_choice].predict(df_encoded)[0]

            # Assuming confidence is returned or calculated by the model
            confidence = 0.85  # Placeholder value

            return jsonify({"Performance Level": prediction, "confidence": confidence})

        else:
            return jsonify({"error": "Request must be in JSON format"}), 400

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
