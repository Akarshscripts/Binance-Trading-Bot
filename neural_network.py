"""
This module contains the neural network built for stock direction classification.
"""

# 3rd party imports
from torch import relu
from torch import nn as NN


class StockModel(NN.Module):
    """
    A neural network model for stock direction prediction using LSTM architecture.

    This model predicts market direction (neutral/up/down) using:
        - LSTM Layer: Multi-layer LSTM for sequential feature extraction.
        - FC Layers: Fully connected layers for classification.
    """

    def __init__(
        self,
        input_size: int,
        lstm_layer_out: int,
        lstm_layer_num: int,
        lstm_dropout: float,
        fc1_out: int,
        fc_dropout: float,
        num_classes: int = 3,
        device: str = "cpu",
    ):
        """
        Initialize the StockModel for direction classification.

        Args:
            input_size (int): Number of input features per timestep.
            lstm_layer_out (int): Hidden size of the LSTM layer.
            lstm_layer_num (int): Number of stacked LSTM layers.
            lstm_dropout (float): Dropout probability between LSTM layers.
            fc1_out (int): Output features of the first FC layer.
            fc_dropout (float): Dropout probability after FC1.
            num_classes (int): Number of direction classes. Defaults to 3.
            device (str): Device to run the model on. Defaults to "cpu".
        """

        super().__init__()

        # LSTM layer
        self.lstm = NN.LSTM(
            input_size=input_size,
            hidden_size=lstm_layer_out,
            num_layers=lstm_layer_num,
            batch_first=True,
            dropout=lstm_dropout,
            device=device,
        )

        # Classification layers
        self.fc1 = NN.Linear(lstm_layer_out, fc1_out, device=device)
        self.dropout = NN.Dropout(fc_dropout)
        self.fc2 = NN.Linear(fc1_out, num_classes, device=device)

    def forward(self, x):
        """
        Forward pass for direction classification.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Direction logits of shape (batch_size, num_classes)
                - Class 0: Sideways/neutral
                - Class 1: Price goes up
                - Class 2: Price goes down
        """

        # LSTM forward
        out, _ = self.lstm(x)

        # take last timestep
        lstm_out = out[:, -1, :]  # (batch, lstm_hidden_size)

        # FC layers
        out = self.fc1(lstm_out)
        out = relu(out)
        out = self.dropout(out)
        direction = self.fc2(out)  # (batch, num_classes)

        return direction
