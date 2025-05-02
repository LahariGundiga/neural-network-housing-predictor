import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Define the activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data                # 8 features
y = data.target.reshape(-1, 1)  # Convert to column vector

# Normalize features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# For faster training, use a small sample of the data
X, _, y, _ = train_test_split(X, y, train_size=500, random_state=42)

# Define the network architecture
input_neurons = X.shape[1]       # 8 features
hidden_neurons = 10              # Number of neurons in the hidden layer
output_neurons = 1               # One output neuron for house price

weights_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))

weights_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Set the training parameters
epochs = 5000
learning_rate = 0.05

# Train the neural network
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, weights_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_output) + bias_output
    predicted_output = sigmoid(final_input)

    # Compute the error (loss)
    error = y - predicted_output
    loss = np.mean(np.square(error))  # Mean Squared Error

    # Backward pass (Backpropagation)
    d_output = error * derivative(predicted_output)
    error_hidden = d_output.dot(weights_output.T)
    d_hidden = error_hidden * derivative(hidden_output)

    # Update weights and biases using gradient descent
    weights_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Occasionally print the loss to track progress
    if (epoch + 1) % 1000 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")

# Display the actual target and final predicted output for the first few data points
print("Actual target (first 5):\n", y[:5])
print("\nFinal predicted output (first 5):\n", predicted_output[:5])

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y, predicted_output)
print(f"Mean Squared Error (MSE): {mse:.6f}")