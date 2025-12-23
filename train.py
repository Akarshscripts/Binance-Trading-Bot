"""
This file is used to train the model using hyperparameter sweep and wandb.
"""

# 1st party imports
from typing import List

# matplotlib backend
import matplotlib

matplotlib.use("Agg")

# 3rd party imports
import wandb
import torch
from torch import nn as NN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# local imports
from neural_network import StockModel
from settings import project_settings
from dataset_handler import DataManager

# --------------------------------------------------
# GLOBAL SPEED SETTINGS
# --------------------------------------------------
DEVICE = project_settings.DEVICE

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# --------------------------------------------------
# SWEEP CONFIG
# --------------------------------------------------
SWEPP_CONFIG = {
    "name": "stock_prediction_sweep_2",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "direction_loss"},
    "parameters": {
        "INPUT_SIZE": {"value": 5},
        "LSTM_HIDDEN_SIZE": {"value": 128},
        "LSTM_NUM_LAYERS": {"value": 2},
        "LSTM_DROPOUT": {"value": 0.2},
        "FC1_OUT_FEATURES": {"value": 64},
        "FC_DROPOUT": {"value": 0.2},
        "OUTPUT_SIZE": {"value": 5},
        "EPOCHS": {"values": [64, 128, 256, 512]},
        "BATCH_SIZE": {"values": [64, 256, 1024]},
        "DIRECTION_LOSS_WEIGHT": {"values": [1, 4, 6]},
        "MAX_LOSS_WEIGHT": {"values": [1, 2]},
        "MIN_LOSS_WEIGHT": {"values": [1, 2]},
        "LEARNING_RATE": {"value": 0.001},
        "FLUCTUATION_LOSS_WEIGHT": {"values": [1, 2, 3]},
        "CROSS_ENTROPY_LOSS_WEIGHTS": {"values": [[1, 3, 3], [2, 6, 6], [2, 8, 8]]},
    },
}


def train_model(
    model: StockModel,
    wandb_run: wandb.Run,
    train_loader: DataLoader,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    regression_criterion: torch.nn.MSELoss,
    classification_criterion: torch.nn.CrossEntropyLoss,
) -> StockModel:
    """
    Train a StockModel using the provided configuration and data.

    Args:
        model: The StockModel instance to train.
        wandb_run: Active wandb run for logging metrics and configuration.
        train_loader: DataLoader containing training batches.
        scaler: GradScaler for mixed precision training.
        optimizer: Optimizer for updating model parameters.
        regression_criterion: MSELoss for max/min price regression.
        classification_criterion: CrossEntropyLoss for direction classification.

    Returns:
        StockModel: The trained model.
    """

    # get the config
    run_config = wandb_run.config

    # train loop
    for epoch in range(run_config.EPOCHS):

        # switch model to training
        model.train()

        # accumulate losses
        acc_max_loss = 0.0
        acc_min_loss = 0.0
        acc_direction_loss = 0.0
        acc_fluctuation_loss = 0.0

        # store total loss
        total_loss = 0.0

        # get bacthes
        for x_batch, y_batch in train_loader:

            # reset the gradient
            optimizer.zero_grad(set_to_none=True)

            # use float16 or float32 based on the device
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):

                # make prediction
                raw_max_pred, raw_min_pred, raw_direction_pred = model(x_batch)

                # extract the final predictions
                max_pred = raw_max_pred[:, -1]
                min_pred = raw_min_pred[:, -1]
                direction_pred = raw_direction_pred[:, -1]

                # calculate losses
                max_loss = regression_criterion(max_pred.squeeze(), y_batch[:, 0, 0])
                min_loss = regression_criterion(min_pred.squeeze(), y_batch[:, 1, 0])
                direction_loss = classification_criterion(
                    direction_pred, y_batch[:, 2, 0].long()
                )

                # calculate the fluctuation between each prediction to make it smoother
                max_diff = raw_max_pred[:, 1:] - raw_max_pred[:, :-1]
                min_diff = raw_min_pred[:, 1:] - raw_min_pred[:, :-1]

                fluctuation_loss = run_config.FLUCTUATION_LOSS_WEIGHT * (
                    regression_criterion(max_diff, torch.zeros_like(max_diff))
                    + regression_criterion(min_diff, torch.zeros_like(min_diff))
                )

                # combine based on weights
                loss = (
                    direction_loss * run_config.DIRECTION_LOSS_WEIGHT
                    + max_loss * run_config.MAX_LOSS_WEIGHT
                    + min_loss * run_config.MIN_LOSS_WEIGHT
                    + fluctuation_loss
                )

            # backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # add loss
            total_loss += loss.item()
            acc_fluctuation_loss += fluctuation_loss.item()
            acc_max_loss += max_loss.item() * run_config.MAX_LOSS_WEIGHT
            acc_min_loss += min_loss.item() * run_config.MIN_LOSS_WEIGHT
            acc_direction_loss += (
                direction_loss.item() * run_config.DIRECTION_LOSS_WEIGHT
            )

        # calculate average loss
        avg_loss = total_loss / len(train_loader)

        # wandb log
        wandb_run.log(
            {
                "avg_loss": avg_loss,
                "train_total_loss": total_loss,
                "train_max_loss": acc_max_loss,
                "train_min_loss": acc_min_loss,
                "train_direction_loss": acc_direction_loss,
                "train_fluctuation_loss": acc_fluctuation_loss,
            }
        )

        # log to console
        print(f"Epoch [{epoch + 1}/{run_config.EPOCHS}] Loss: {avg_loss:.6f}")

    # return the model
    return model


def evaluate_model(
    model: StockModel,
    testX: torch.Tensor,
    batch_size: int,
) -> tuple[list, list, list]:
    """
    Evaluate a StockModel on test data and generate predictions.

    Args:
        model: The StockModel instance to evaluate.
        testX: Test input tensor.
        batch_size: Batch size for inference.

    Returns:
        tuple: (max_preds, min_preds, direction_preds) as lists.
    """

    # switch model for inference
    model.eval()

    # hold predicted values
    max_preds, min_preds, direction_preds = [], [], []

    # with no gradient
    with torch.no_grad():

        # iterate
        for i in range(0, len(testX), batch_size):

            # get current batch
            x_batch = testX[i : i + batch_size]

            # predict
            max_pred, min_pred, direction_pred = model(x_batch)

            # we want the last timestep prediction for each sequence in the batch
            # max_pred shape: (batch_size, sequence_length)
            # we take the last timestep: max_pred[:, -1]
            # this gives us (batch_size,) for max and min predictions
            max_pred_last = max_pred[:, -1].cpu()
            min_pred_last = min_pred[:, -1].cpu()
            direction_pred_last = direction_pred[:, -1].cpu()

            # append
            max_preds.extend(max_pred_last.tolist())
            min_preds.extend(min_pred_last.tolist())
            direction_preds.extend(direction_pred_last.argmax(dim=1).tolist())

    # return the predictions
    return max_preds, min_preds, direction_preds


def create_confusion_matrix(
    actual_directions: List[int], direction_preds: List[int]
) -> plt.Figure:
    """
    Create and display a confusion matrix for direction predictions.

    Labels:
    0 = Neutral, 1 = Up, 2 = Down

    Args:
        actual_directions: List of actual direction labels.
        direction_preds: List of predicted direction labels.

    Returns:
        plt.Figure: The confusion matrix figure.
    """

    # class names
    class_names = ["Neutral", "Up", "Down"]

    # force fixed class order
    cm = confusion_matrix(actual_directions, direction_preds, labels=[0, 1, 2])

    # create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    # colorbar
    fig.colorbar(im, ax=ax)

    # ticks & tick labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Direction Confusion Matrix")

    # write values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    # layout
    plt.tight_layout()
    return fig


def run_sweep():
    """
    Run a hyperparameter sweep experiment using wandb.

    This function is called by wandb.agent to train and evaluate a model
    with hyperparameters specified in the sweep configuration.

    Uses global trainX, trainY, testX, testY tensors for training and evaluation.
    """

    global trainX, trainY, testX, testY

    # init wandp
    wandb_run = wandb.init(reinit="finish_previous")
    sweep_config = wandb_run.config

    # log the config once
    wandb.log(
        {
            "INPUT_SIZE": wandb_run.config.INPUT_SIZE,
            "LSTM_HIDDEN_SIZE": sweep_config.LSTM_HIDDEN_SIZE,
            "LSTM_NUM_LAYERS": sweep_config.LSTM_NUM_LAYERS,
            "LSTM_DROPOUT": sweep_config.LSTM_DROPOUT,
            "FC1_OUT_FEATURES": sweep_config.FC1_OUT_FEATURES,
            "FC_DROPOUT": sweep_config.FC_DROPOUT,
            "OUTPUT_SIZE": sweep_config.OUTPUT_SIZE,
            "EPOCHS": sweep_config.EPOCHS,
            "BATCH_SIZE": sweep_config.BATCH_SIZE,
            "DIRECTION_LOSS_WEIGHT": sweep_config.DIRECTION_LOSS_WEIGHT,
            "MAX_LOSS_WEIGHT": sweep_config.MAX_LOSS_WEIGHT,
            "MIN_LOSS_WEIGHT": sweep_config.MIN_LOSS_WEIGHT,
            "LEARNING_RATE": sweep_config.LEARNING_RATE,
            "FLUCTUATION_LOSS_WEIGHT": sweep_config.FLUCTUATION_LOSS_WEIGHT,
            "CROSS_ENTROPY_LOSS_WEIGHTS": sweep_config.CROSS_ENTROPY_LOSS_WEIGHTS,
        }
    )

    # setup training dataset
    train_dataset = TensorDataset(trainX, trainY)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sweep_config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # create the model
    model = StockModel(
        sweep_config.INPUT_SIZE,
        sweep_config.LSTM_HIDDEN_SIZE,
        sweep_config.LSTM_NUM_LAYERS,
        sweep_config.LSTM_DROPOUT,
        sweep_config.FC1_OUT_FEATURES,
        sweep_config.FC_DROPOUT,
        sweep_config.OUTPUT_SIZE,
        device=DEVICE,
    )

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=sweep_config.LEARNING_RATE)

    # convert array to tensor
    CROSS_ENTROPY_LOSS_WEIGHTS = torch.tensor(
        sweep_config.CROSS_ENTROPY_LOSS_WEIGHTS, dtype=torch.float, device=DEVICE
    )

    # create loss functions for regression (max/min) and classification (direction)
    regression_criterion = NN.MSELoss()
    classification_criterion = NN.CrossEntropyLoss(weight=CROSS_ENTROPY_LOSS_WEIGHTS)

    # create a scaler
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # train the model
    model = train_model(
        model=model,
        scaler=scaler,
        optimizer=optimizer,
        wandb_run=wandb_run,
        train_loader=train_loader,
        regression_criterion=regression_criterion,
        classification_criterion=classification_criterion,
    )

    # evaluate the model
    max_preds, min_preds, direction_preds = evaluate_model(
        model=model,
        testX=testX,
        batch_size=sweep_config.BATCH_SIZE,
    )

    # compare the direction preds and calculate accuracy
    actual_directions = testY[:, 2, 0].cpu().tolist()
    correct = sum(p == a for p, a in zip(direction_preds, actual_directions))
    direction_accuracy = correct / len(actual_directions)

    # create confusion matrix
    confusion_matrix = create_confusion_matrix(actual_directions, direction_preds)

    # get the actual max and min preds
    actual_max_preds = testY[:, 0, 0].cpu().tolist()
    actual_min_preds = testY[:, 1, 0].cpu().tolist()

    # shorten the arrays to 5000 for logging (wandb restrictions)
    if len(actual_min_preds) >= 5000:
        max_preds = max_preds[:5000]
        min_preds = min_preds[:5000]
        actual_max_preds = actual_max_preds[:5000]
        actual_min_preds = actual_min_preds[:5000]

    # xs for line series
    xs = list(range(len(max_preds)))

    # log to wandb and upload graphs
    wandb_run.log(
        {
            "direction_accuracy": direction_accuracy,
            "confusion_matrix": wandb.Image(confusion_matrix),
            "max_pred_plot": wandb.plot.line_series(
                xs=xs,
                ys=[max_preds, actual_max_preds],
                keys=["predicted", "actual"],
                title="Actual vs Predicted Maximum Price",
            ),
            "min_pred_plot": wandb.plot.line_series(
                xs=xs,
                ys=[min_preds, actual_min_preds],
                keys=["predicted", "actual"],
                title="Actual vs Predicted Minimum Price",
            ),
        }
    )


if __name__ == "__main__":

    # login
    wandb.login()

    # the csv file name
    csv_file_name = "XRP_USDT.csv"
    dataframe_manager = DataManager(
        csv_file=csv_file_name, device=DEVICE, train_ratio=0.9
    )

    # the columns to use
    feature_cols = ["open", "EMA_0_diff", "EMA_1_diff", "ADX_2", "RSI_3"]
    target_col = ["future_max", "future_min", "label"]

    # get training and testing tensors
    trainX, trainY = dataframe_manager.get_train_tensors(
        feature_col=feature_cols, target_col=target_col
    )
    testX, testY = dataframe_manager.get_test_tensors(
        feature_col=feature_cols, target_col=target_col
    )

    # create sweep
    # sweep_id = wandb.sweep(SWEPP_CONFIG, project="stock-prediction")
    # print("Sweep ID: ", sweep_id)

    # run sweep
    sweep_id = "4m60hnex"
    wandb.agent(
        sweep_id=sweep_id, function=run_sweep, project="stock-prediction", count=20
    )
