"""
This module contains the core logic to load a model and use it for inference.
The model is expected to be a torch.jit.scripted model and the config is expected to be a json file.

The model config is expected to have the following keys:
- input_size: The input size of the model.
- output_size: The output size of the model.
- input_shape: The input shape of the model.
- sequence_length: The sequence length of the model.
"""

# 1st party imports
from typing import Literal

# 3rd party imports
import torch

# local imports
from settings import Constants


class NnModel:
    """
    Wrapper class for loading and running inference on PyTorch models.

    Attributes:
        device (str): The device to run inference on ('cuda' or 'cpu').
        model (torch.nn.Module): The loaded PyTorch model.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the NnModel with a saved checkpoint.

        Args:
            model_path (str): Path to the saved model checkpoint.
            device (str): Device to load the model on. Defaults to 'cuda'.
                Falls back to 'cpu' if CUDA is not available.
        """

        # set device
        self.device = device if torch.cuda.is_available() else "cpu"

        # load model
        self.model = self.__load_model(model_path)

    def __load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load a model from a checkpoint file.

        Args:
            model_path (str): Path to the saved model checkpoint.

        Returns:
            torch.nn.Module: The loaded model ready for inference.

        Raises:
            ValueError: If checkpoint format is invalid.
        """

        # load the data
        model = torch.jit.load(model_path, map_location=self.device)
        model.to(self.device)
        model.eval()

        # return the model
        return model

    def get_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run inference on the model.

        Args:
            inputs (torch.Tensor): Input tensor for the model.

        Returns:
            torch.Tensor: Model output tensor.
        """

        # move inputs to device
        inputs = inputs.to(self.device)

        # run inference without gradient computation
        with torch.no_grad():
            outputs = self.model(inputs)

        # return the outputs
        return outputs

    def convert_preds(self, preds: torch.Tensor) -> Literal["neutral", "up", "down"]:
        """
        Convert raw model predictions to meaningful class labels.

        This method transforms the model's output logits into human-readable
        class predictions. The logic may be updated in the future as the
        prediction interpretation evolves.

        Args:
            preds (torch.Tensor): Raw model output tensor of shape (1, num_classes)
                containing logits for each class.

        Returns:
            str: The predicted class label from Constants.OUT_CLASS
                (e.g., 'neutral', 'up', 'down').
        """

        # get the argmax
        preds = preds.argmax(dim=1).cpu().item()

        # return the preds
        return Constants.OUT_CLASS[preds]
