import joblib
from sklearn.metrics import mean_squared_error
from utils import get_regression_data

def test_regression_model():
    # Load the test data
    _, X_test, _, y_test = get_regression_data()
    
    # Load the saved regression model
    model = joblib.load("models/regression_model.pkl")
    
    # Generate predictions and compute the Mean Squared Error (MSE)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Regression Model Test MSE: {mse:.4f}")

if __name__ == "__main__":
    test_regression_model()

