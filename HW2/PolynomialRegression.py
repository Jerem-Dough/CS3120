# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(5)

# Instantiate
m = 500
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**5 - 5 * X**3 - X**2 + 2 + 5 * np.random.randn(m, 1)

# Split Data into Training and Testing Sets
train_size = 300
X_train, X_test = X[:train_size], X[train_size:]  # Training and testing set
y_train, y_test = y[:train_size], y[train_size:]

# Initialize Polynomial Regression Parameters for storing and testing purposes
degrees = range(2, 26)
train_errors = []
test_errors = []
r2_scores = []

# Selected degrees for visualization
selected_degrees = [2, 5, 8, 10, 20]
selected_plots = {}

# Define X for plotting
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

# Training Polynomial Models and Computing Metrics
for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False) # Transform X values into polynomial features
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.transform(X_test)
    X_poly_plot = poly_features.transform(X_plot)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    # Generate predictions
    y_train_pred = model.predict(X_poly_train)
    y_test_pred = model.predict(X_poly_test)
    y_plot_pred = model.predict(X_poly_plot)

    # Compute Mean Squared Error (MSE) for training and testing sets
    train_loss = mean_squared_error(y_train, y_train_pred)
    test_loss = mean_squared_error(y_test, y_test_pred)

    # Compute R^2 score for model evaluation
    r2 = r2_score(y_test, y_test_pred)

    # Store computed values
    train_errors.append(train_loss)
    test_errors.append(test_loss)
    r2_scores.append(r2)

    # Store predictions for selected degrees
    if degree in selected_degrees:
        selected_plots[degree] = (X_plot, y_plot_pred)

# Plotting Polynomial Fits for Selected Degrees
plt.figure(figsize=(12, 8))

for i, degree in enumerate(selected_degrees, 1):
    X_plot, y_plot_pred = selected_plots[degree]

    plt.subplot(2, 3, i)
    plt.scatter(X_train, y_train, color='green', alpha=0.5, label="Training Data")
    plt.plot(X_plot, y_plot_pred, color='blue', linewidth=2, label=f"Degree {degree}")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.title(f"Polynomial Regression with Degree: {degree}")
    plt.legend()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

plt.show()

# Polynomial_Regression_by_Degree.png

# Plotting Training and Testing Loss Across Degrees
plt.figure(figsize=(8, 6))
plt.plot(degrees, train_errors, label="Training Loss", marker='o', color='green', linewidth=2)
plt.plot(degrees, test_errors, label="Testing Loss", marker='s', color='blue', linewidth=2)
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Testing Loss")
plt.legend()
plt.show()

# Training_vs_Testing_Lost.png

# Plot R^2 Scores Across Degrees
plt.figure(figsize=(8, 6))
plt.plot(degrees, r2_scores, marker='o', linestyle='-', color='orange', linewidth=2)
plt.xlabel("Polynomial Degree")
plt.ylabel("R^2 Score")
plt.title("Model Performance (R^2 Score)")
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.show()

# Model_Performance.png

# From the plots it is fairly clear that a polynomial degree of 8 is ideal. 
# If we go with a higher-degree model, we're trending towards overfitting the training data. 
# Many of the lower-degree models tend to underfit the data. 
# We can also see issues like bias and variance among these other degrees. 
# Degree 8 appears to be minimizing bias, variance, and model fitting issues.