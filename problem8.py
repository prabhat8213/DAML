# Program 8: KNN Classifier on Iris Dataset

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize KNN Classifier (default k=5)
knn = KNeighborsClassifier()

# Train the model
knn.fit(x_train, y_train)

# Predict on test set
y_pred = knn.predict(x_test)

# Print predicted values
print("Predicted values:\n", y_pred)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print results
print("\nEvaluation Metrics:")
print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)
