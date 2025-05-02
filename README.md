# Neural Network Housing Price Predictor

This project implements a simple neural network to predict housing prices based on the California Housing dataset using backpropagation. The model uses the sigmoid activation function and is trained using gradient descent.

## Overview

This neural network model predicts housing prices by using features such as median income, house age, and the number of rooms in a district. The training process involves forward propagation, backpropagation, and gradient descent to adjust the model's weights.

## Dataset

The model uses the **California Housing dataset** from the `sklearn.datasets` module. This dataset contains information about various attributes of houses in California, such as:

- Median income
- House age
- Average number of rooms
- Average number of bedrooms
- Population
- Households
- Latitude and longitude

## Installation

1. Clone the repository:

    git clone https://github.com/LahariGundiga/neural-network-housing-predictor.git

2. Install the required libraries:

    pip install numpy scikit-learn

## Usage

Run the neural_network.py script to train the neural network and predict housing prices:

  python neural_network.py

The model will train over 5000 epochs and print the loss at every 1000 epochs. After training, it will output the final predicted housing prices and the actual values for comparison.

## Training Process

1. Data Preprocessing: The features and target variable are normalized to the range [0, 1] using MinMaxScaler.

2. Forward Pass: The data is passed through the network layers to produce predictions.

3. Error Calculation: The error is computed using Mean Squared Error (MSE).

4. Backpropagation: The gradients are calculated and weights are updated using gradient descent.

5. Epochs: The network is trained over 5000 epochs.

## Evaluation

After training, the model's performance is evaluated using Mean Squared Error (MSE), and the predicted housing prices are compared with the actual values for the first 5 data points.

## Author

Lahari Gundiga

GitHub: [LahariGundiga](https://github.com/LahariGundiga)
