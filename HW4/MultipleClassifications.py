# Import required libraries
import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)
iris = pd.read_csv('/content/drive/My Drive/IRIS.csv')

# Convert strings to expected values
encoder = LabelEncoder()
iris['species'] = encoder.fit_transform(iris['species'])

# Set features and target
X = iris.drop('species', axis=1) # Features
y = iris['species'] # Predictions/Targets

# Split data into 3 categories, 70% for training
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% for testing
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state=42)  # 10% for validation

# Scale / Normalize values (Standard Deviation = 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize 3 different classification algorithms
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
knn = KNeighborsClassifier(n_neighbors=6)
lr = LogisticRegression(max_iter=200)

# Train each of our classification algorithms on our given dataset
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Create method to be used for metrics
def get_classification_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, average='macro'),
        'Recall': recall_score(y_test, predictions, average='macro'),
        'F1 Score': f1_score(y_test, predictions, average='macro')
    }

# Evaluate models and get metrics
svm_result = get_classification_metrics(svm, X_test, y_test)
knn_result = get_classification_metrics(knn, X_test, y_test)
lr_result = get_classification_metrics(lr, X_test, y_test)

# Display results
print("Support Vector Machine (SVM):", svm_result)
print("K-Nearest Neighbors (KNN):", knn_result)
print("Logistic Regression:", lr_result)

# Combine all results into a Python dictionary
results = {
    'SVM': svm_result,
    'KNN': knn_result,
    'Logistic Regression': lr_result
}

# Create single variable, defining all metrics we want to use
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Create evenly spaced x-axis labels
x = np.arange(len(metrics))
width = 0.25

# Plot and display our data using Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
cool_colors = ['#4A90E2', '#50E3C2', '#9013FE']

for i, (model_name, scores) in enumerate(results.items()):
    values = [scores[metric] for metric in metrics]
    ax.bar(x + i * width, values, width=width, label=model_name, color = cool_colors[i])

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparisons')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0.9, 1.05)
ax.legend()
plt.show()