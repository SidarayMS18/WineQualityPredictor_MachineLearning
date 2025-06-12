import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Paths to saved models
regression_model_path = "models/regression_model.pkl"
classification_model_path = "models/classification_model.pkl"

# Load models if they exist
regression_model = joblib.load(regression_model_path) if os.path.exists(regression_model_path) else None
classification_model = joblib.load(classification_model_path) if os.path.exists(classification_model_path) else None

@app.route('/')
def index():
    # Render the homepage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form input and cast to float.
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])
        model_type = request.form['model_type']  # Expected to be 'regression' or 'classification'

        # Assemble features in the expected order
        features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                              chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                              pH, sulphates, alcohol]])
    except Exception as e:
        return jsonify({'error': f'Error processing input: {str(e)}'})

    if model_type == 'regression':
        if regression_model is None:
            return jsonify({'error': 'Regression model not found. Please train it first.'})
        prediction = regression_model.predict(features)
        result = float(prediction[0])
        return render_template('result.html', model_type='Regression', result=result)
    elif model_type == 'classification':
        if classification_model is None:
            return jsonify({'error': 'Classification model not found. Please train it first.'})
        prediction = classification_model.predict(features)
        label = int(prediction[0])
        quality = 'High Quality' if label == 1 else 'Low Quality'
        return render_template('result.html', model_type='Classification', result=quality)
    else:
        return jsonify({'error': 'Invalid model type provided.'})

if __name__ == "__main__":
    app.run(debug=True)
