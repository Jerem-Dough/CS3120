# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Instantiate arrays with required values.
x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
y_data= np.array([129.54611622, 135.54611622, 121.54611622, 99.54611622, 103.54611622, 109.54611622, 93.54611622, 179.54611622, 75.54611622, 179.54611622])

bb = np.arange(0, 100, 1)  # Bias values
ww = np.arange(-5, 5, 0.1) # Weight values
Z = np.zeros((len(ww), len(bb))) # Loss matrix

# Compute the loss for each combination of w and b
for i in range(len(bb)):
  for j in range(len(ww)):
    b = bb[i]
    w = ww[j]
    Z[j][i] = np.mean((w * x_data + b - y_data) ** 2)

# Plot and display the landscape of the loss function
plt.figure(figsize = (8,6))
plt.contourf(bb, ww, Z, levels = 25, cmap = "jet")
plt.xlabel("Bias")
plt.ylabel("Weight")
plt.title("Loss Function Landscape")
plt.show()

# Loss_Function_Landscape.png

b = 0  # Initial bias
w = 0  # Initial weight
lr = 0.0004 # Learning rate (Explanation in bottom code block)
iterations = 13000 # Iterations for gradient descent (Explanation in bottom code block)

# Store history to display and plot
b_history = [b]
w_history = [w]

# Begin gradient descent loop
for i in range(iterations):
  prediction = w * x_data + b # Compute predictions using current weight and bias
  error = prediction - y_data # Compute error using our prediction and actual value

  w_gradient = (2 / len(x_data)) * np.dot(error, x_data) # Compute the gradient of the loss function with respect to w using dot product of error and input values
  b_gradient = (2 / len(x_data)) * np.sum(error) # Compute the gradient of the loss function with respect to b by summing errors across all data points

  w -= lr * w_gradient # Update to new optimized weight value
  b -= lr * b_gradient # Update to new optimized bias value

  w_history.append(w) # Store weight for future use
  b_history.append(b) # Store bias for future use

# Plot and display our weight and bias changes as we iterate
plt.figure(figsize = (8,6))
plt.contourf(bb, ww, Z, levels=50, cmap="jet")
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.scatter(b_history[-1], w_history[-1], color='orange', s=100, marker='x')
plt.xlabel("Bias")
plt.ylabel("Weight")
plt.title("Tracking Changes of Weights and Biases in Gradient Descent")
plt.show()

# Tracking_Parameter_Changes.png

# Compute final predictions
prediction_final = w * x_data + b

# Compute Mean Squared Error
mse = np.mean((y_data - prediction_final) ** 2)
print(f"Mean Squared Error: {mse:.4f}")

# Plot and display predictions vs. actual values
plt.figure(figsize=(8,6))
plt.scatter(y_data, prediction_final, color='blue')
plt.plot(y_data, y_data, color='black')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Comparison of Model Predictions and True Values")
plt.show()

# Predictions_vs_Truth_Comparisons.png

# Learning rate controls the step size for weight and bias updates in gradient descent.
  # A smaller learning rate ensures slow and steady convergence, reducing the risk of overshooting.
  # A larger learning rate speeds up convergence but can cause instability or divergence.
  # I chose a small learning rate to ensure stability, especially given our small dataset.

# The number of iterations determines how many times gradient descent updates our weights and biases.
  # More iterations allow gradient descent to converge closer to the minimum loss value.
  # Too few iterations may stop before reaching an optimal solution.
  # Too many iterations can lead to unnecessary computations without significant improvement.

# This implementation of gradient descent terminates after a fixed number of iterations.
  # This prevents the training process from running indefinitely.
  # After testing various iteration counts with a learning rate of 0.0004, I found that 13,000 iterations brought us very close to the optimal minimum.
  # Using a fixed iteration count for termination allowed me to adjust a single value while testing different gradient descent configurations.