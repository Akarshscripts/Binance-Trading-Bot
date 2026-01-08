"""
This module contains the neural network built for stock prediction.

The StockModel class implements a feedforward neural network for stock price prediction
using fully connected layers with ReLU activation and dropout regularization.
"""

# 3rd party imports
from torch import relu
from torch import nn as NN


class StockModel(NN.Module):
    """
    A neural network model for stock price prediction using feedforward architecture.

    This model predicts stock prices using:
        - FC Layers: Fully connected layers with ReLU activation.
        - Dropout: Regularization to prevent overfitting.
        - Linear Output: Single output value for price prediction.
    """

    def __init__(
        self,
        fc1_input_size: int,
        fc2_input_size: int,
        dropout: float,
        device: str = "cpu",
    ):
        """
        Initialize the StockModel for stock price prediction.

        Args:
            fc1_input_size (int): Number of neurons in the first hidden layer.
            fc2_input_size (int): Number of neurons in the second hidden layer.
            dropout (float): Dropout probability for regularization.
            device (str): Device to run the model on. Defaults to "cpu".
        """

        super().__init__()

        # layers
        self.fc0 = NN.LazyLinear(fc1_input_size, device=device)
        self.fc1 = NN.Linear(fc1_input_size, fc2_input_size, device=device)
        self.dropout = NN.Dropout(dropout)
        self.fc2 = NN.Linear(fc2_input_size, 1, device=device)

    def forward(self, x):
        """
        Forward pass for stock price prediction.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Predicted stock price of shape (batch_size, 1)
        """

        x = relu(self.fc0(x))
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
