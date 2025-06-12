import pandas as pd
from sklearn.model_selection import train_test_split

def load_wine_data(url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"):
    """
    Load the wine quality dataset from the provided URL.
    The CSV file is semicolon separated.
    """
    df = pd.read_csv(url, sep=";") 
    df.shape
    return df

def get_regression_data(test_size=0.2, random_state=42):
    """
    Splits the wine dataset for regression.
    
    Features:
      All columns except 'quality'.
      
    Target:
      The 'quality' column.
      
    Returns:
      X_train, X_test, y_train, y_test
    """
    df = load_wine_data()
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_classification_data(test_size=0.2, random_state=42):
    """
    Splits the wine dataset for classification.
    
    Converts the 'quality' score into a binary target:
      - 0 (low quality) for quality <= 5
      - 1 (high quality) for quality >= 6
      
    Returns:
      X_train, X_test, y_train, y_test
    """
    df = load_wine_data()
    # Create a binary label based on wine quality
    df["quality_label"] = (df["quality"] >= 6).astype(int)
    
    # Drop columns related to original quality to get features
    X = df.drop(["quality", "quality_label"], axis=1)
    y = df["quality_label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

