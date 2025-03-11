# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Load dataset from Google Drive
file_path = r'/content/drive/MyDrive/pima-indians-diabetes-database.csv'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(file_path, header=None, names=col_names)

# Select important features based on relevance
col_names = ['pregnant', 'glucose', 'bmi', 'age', 'pedigree']
X = pima[col_names]
y = pima['label']

# Replace bad data (0 values in columns that shouldn't be zero)
def replace_bad_data(df, columns):
    for col in columns:
        if col in df.columns:  # Ensure column exists in DataFrame before modifying
            df.loc[:, col] = df.loc[:, col].astype(float)  # Ensure float dtype before replacing NaN (Not a number / inaccurate and unrepresented data)
            df.loc[:, col] = df.loc[:, col].replace(0, np.nan)
            df.loc[:, col] = df.loc[:, col].fillna(df[col].median())
    print("Missing values after replacement:")
    print(df.isna().sum())  # Debugging step to verify missing values. This step is useful for new data sets.
    return df

# Feature scaling with Min-Max Normalization
def feature_scaling(df):
    mins = df.min()
    maxes = df.max()
    return (df - mins) / (maxes - mins)

X = feature_scaling(X)

# Split dataset: 60% training / 40% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train model using logistic regression... with only two lines of code!
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_prediction = model.predict(X_test)
y_prediction_probability = model.predict_proba(X_test)[:, 1]

# Calculate various evaluatiom metrics
conf_matrix = confusion_matrix(y_test, y_prediction)
precision = precision_score(y_test, y_prediction)
recall = recall_score(y_test, y_prediction)
f1 = f1_score(y_test, y_prediction)
roc_auc = roc_auc_score(y_test, y_prediction_probability)

# Print evaluation metrics
print("Confusion Matrix:\n", conf_matrix)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot and display ROC Curve, labeled accordingly.
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', label=f'data 1, AUC = {roc_auc:.2f}')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(y_test, y_prediction_probability)