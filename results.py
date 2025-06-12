import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import precision_recall_curve, average_precision_score
from utils import get_classification_data

# Load classification test data
_, X_test_clf, _, y_test_clf = get_classification_data()
# Load the trained classification model
classification_model = joblib.load("models/classification_model.pkl")
# Get predicted probabilities for the positive class (assumes positive class is labeled 1)
y_scores = classification_model.predict_proba(X_test_clf)[:, 1]

# Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test_clf, y_scores)
avg_precision = average_precision_score(y_test_clf, y_scores)

# Plot Precision-Recall Curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, marker=".", label=f"AP: {avg_precision:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Classification Model)")
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import joblib
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from utils import get_regression_data, get_classification_data

# ---------------------
# Regression Evaluation
# ---------------------
# Load regression test data
_, X_test_reg, _, y_test_reg = get_regression_data()
# Load the trained regression model
regression_model = joblib.load("models/regression_model.pkl")
# Predict on test data
y_pred_reg = regression_model.predict(X_test_reg)
# Calculate Mean Squared Error
mse = mean_squared_error(y_test_reg, y_pred_reg)

# ------------------------
# Classification Evaluation
# ------------------------
# Load classification test data
_, X_test_clf, _, y_test_clf = get_classification_data()
# Load the trained classification model
classification_model = joblib.load("models/classification_model.pkl")
# Predict on test data (binary predictions)
y_pred_clf = classification_model.predict(X_test_clf)
# Calculate Accuracy
accuracy = accuracy_score(y_test_clf, y_pred_clf)
# Build Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)

# ------------------------
# Plot the comparison graphs
# ------------------------
plt.figure(figsize=(14, 6))

# Subplot 1: Regression: Actual vs Predicted Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6, edgecolors="k")
plt.xlabel("Actual Wine Quality")
plt.ylabel("Predicted Wine Quality")
plt.title(f"Regression: Actual vs Predicted\nMSE: {mse:.2f}")
# Plot the ideal prediction line (diagonal)
min_val = min(np.min(y_test_reg), np.min(y_pred_reg))
max_val = max(np.max(y_test_reg), np.max(y_pred_reg))
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

# Subplot 2: Classification: Confusion Matrix
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"Classification: Confusion Matrix\nAccuracy: {accuracy:.2f}")

plt.tight_layout()
plt.show()
