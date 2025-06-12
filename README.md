# Wine Quality Prediction Project

This project uses the UCI Wine Quality (Red) dataset to predict the quality of wine based on its chemical features. Two machine learning approaches are provided:

1. **Linear Regression (Regression):**
   - **Goal:** Predict the wine quality as a continuous variable.
   - **Algorithm:** Linear Regression.
   - **Dataset:** UCI Wine Quality Dataset (red wine).
   - **Usage:** Run `train_regression.py` to train and then `test_regression.py` to evaluate.

2. **Random Forest (Classification):**
   - **Goal:** Predict wine quality as a binary class.
   - **Method:** Convert quality scores to binary labels (0 for low quality if quality ≤ 5, 1 for high quality if quality ≥ 6).
   - **Algorithm:** Random Forest Classifier.
   - **Usage:** Run `train_classification.py` to train and then `test_classification.py` to evaluate.

## New Feature: Flask Web Interface

A Flask interface (`app.py`) has been added that provides a simple web form. Users can enter the 11 chemical features and select a model type (Regression or Classification) to obtain a prediction.

## Project Structure


## Dataset

The dataset is available at:  
[Wine Quality (Red) Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)  
It is a semicolon-separated CSV file that includes several chemical features (e.g., acidity, sugar, pH) along with a quality score.

## Prerequisites

- Python 3.7 or higher  
- pip package installer

## Installation

1. Clone the repository or copy the project files.
2. Navigate to the project directory and install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Training and Testing the Models

### Regression with Linear Regression

1. **Train the Regression Model:**
    ```bash
    python train_regression.py
    ```
2. **Test the Regression Model:**
    ```bash
    python test_regression.py
    ```

### Classification with Random Forest

1. **Train the Classification Model:**
    ```bash
    python train_classification.py
    ```
2. **Test the Classification Model:**
    ```bash
    python test_classification.py
    ```

## Running the Flask Web Interface

1. **Start the Flask Server:**
    ```bash
    python app.py
    ```
2. **Access the Interface:**
   Open your browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the prediction form.

Fill in the wine's chemical features, select the model type (Regression or Classification), and click **Predict**. The result page will display either the predicted wine quality (for regression) or the quality label (for classification).

Feel free to extend or modify the code for your specific needs.
