import joblib
from sklearn.metrics import accuracy_score
from utils import get_classification_data

def test_classification_model():
    # Load test data for classification
    _, X_test, _, y_test = get_classification_data()
    
    # Load the saved classification model
    model = joblib.load("models/classification_model.pkl")
    
    # predictions and compute the accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Classification Model Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_classification_model()
