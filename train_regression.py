import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import get_regression_data

def train_regression_model():
    # Load training and testing data
    X_train, X_test, y_train, y_test = get_regression_data()
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model on test data
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Regression Model trained. Test MSE: {mse:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/regression_model.pkl")
    print("Model saved to models/regression_model.pkl")

if __name__ == "__main__":
    train_regression_model()

