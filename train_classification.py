import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import get_classification_data

def train_classification_model():
    # Load training and testing data for classification
    X_train, X_test, y_train, y_test = get_classification_data()
    
    # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on test data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Classification Model trained. Test Accuracy: {accuracy:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/classification_model.pkl")
    print("Model saved to models/classification_model.pkl")

if __name__ == "__main__":
    train_classification_model()

