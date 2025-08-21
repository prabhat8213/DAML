# Program 6: Multiple Linear Regression (MLR) on Iris Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the dataset
iris = pd.read_csv("IRIS.csv")
print("First 5 rows of dataset:")
print(iris.head(), "\n")

# 2. Define independent (X) and dependent (y) variables
X = iris[['SepalLengthCm', 'SepalWidthCm']]
y = iris['PetalLengthCm']

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Linear Regression model
LR = LinearRegression()
LR.fit(X_train, y_train)

# 5. Predict on test data
y_pred = LR.predict(X_test)

print("Predicted values:\n", y_pred, "\n")

# 6. Evaluate the model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 7. Visualization - Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Actual vs Predicted Petal Length (MLR)")
plt.show()
