"""
This module contains the neural network built for stock price prediction
"""

# 3rd party imports
from torch import relu
from torch import nn as NN


class StockModel(NN.Module):
    """
    A neural network model for stock price prediction using LSTM architecture.

    This model combines LSTM layers with fully connected layers to predict:
    - Maximum price in the prediction window
    - Minimum price in the prediction window
    - Direction label (down/neutral/up)

    Architecture:
        - LSTM Layer: Multi-layer LSTM for sequential feature extraction.
        - FC1: Linear layer with ReLU activation.
        - Dropout: Dropout layer for regularization.
        - FC3: Output layer producing max, min, and direction predictions.

    Attributes:
        lstm (nn.LSTM): LSTM layer for sequential feature extraction.
        fc1 (nn.Linear): First fully connected layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        fc3 (nn.Linear): Output layer producing prediction values.
    """

    def __init__(
        self,
        input_size: int,
        lstm_layer_out: int,
        lstm_layer_num: int,
        lstm_dropout: float,
        fc1_out: int,
        fc_dropout: float,
        output_size: int,
        device: str = "cpu",
    ):
        """
        Initialize the StockModel with LSTM and fully connected layers.

        Args:
            input_size (int): Number of input features per timestep.
            lstm_layer_out (int): Hidden size of the LSTM layer.
            lstm_layer_num (int): Number of stacked LSTM layers.
            lstm_dropout (float): Dropout probability between LSTM layers.
            fc1_out (int): Output features of the first fully connected layer.
            fc_dropout (float): Dropout probability after FC1.
            output_size (int): Number of output features (max, min, direction classes).
            device (str): Device to run the model on. Defaults to "cpu".
        """

        # super init
        super().__init__()

        # create the LSTM layer
        self.lstm = NN.LSTM(
            input_size=input_size,
            hidden_size=lstm_layer_out,
            num_layers=lstm_layer_num,
            batch_first=True,
            dropout=lstm_dropout,
            device=device,
        )

        # create the 1st FC layer
        self.fc1 = NN.Linear(
            in_features=lstm_layer_out,
            out_features=fc1_out,
            device=device,
        )

        # create the dropout layer
        self.dropout = NN.Dropout(fc_dropout)

        # create the final classification layer
        self.fc3 = NN.Linear(
            in_features=fc1_out,
            out_features=output_size,
            device=device,
        )

    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
                containing the historical stock data features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - max_val: Tensor of shape (batch_size, 1) with predicted maximum price.
                - min_val: Tensor of shape (batch_size, 1) with predicted minimum price.
                - label: Tensor of shape (batch_size, num_classes) with direction logits.
                    - Class 0: Sideways/neutral
                    - Class 1: Price goes up
                    - Class 2: Price goes down
        """

        # pass the lstm layer
        out, _ = self.lstm(x)

        # take last timestep
        # out = out[:, -1, :]  # (batch, 64)

        # pass through 1st FC layer
        out = self.fc1(out)
        out = relu(out)

        # pass through the dropout
        out = self.dropout(out)

        # pass through the final FC layer
        out = self.fc3(out)  # (batch, 3)

        # extract outputs (max, min, label)
        max_val = out[:, :, 0]
        min_val = out[:, :, 1]
        label = out[:, :, 2:]

        # return the max, min and label
        # (batch_size, timestamps, 1)
        return max_val, min_val, label
