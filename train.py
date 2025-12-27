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
from dataset_handler import DataManager

# --------------------------------------------------
# GLOBAL SPEED SETTINGS
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
print("Using device: ", DEVICE)


# --------------------------------------------------
# SWEEP CONFIG
# --------------------------------------------------
SWEPP_CONFIG = {
    "name": "direction_only_sweep_2",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "direction_accuracy"},
    "parameters": {
        "INPUT_SIZE": {"value": 8},
        "LSTM_HIDDEN_SIZE": {"values": [32, 64, 128, 256]},
        "LSTM_NUM_LAYERS": {"values": [2, 3]},
        "LSTM_DROPOUT": {"values": [0.3, 0.4, 0.5]},
        "FC1_OUT_FEATURES": {"values": [16, 32, 64, 128]},
        "FC_DROPOUT": {"values": [0.4, 0.5, 0.6]},
        "NUM_CLASSES": {"value": 3},
        "VAL_PATIENCE": {"value": 10},
        "EPOCHS": {"values": [50, 100, 150]},
        "BATCH_SIZE": {"values": [32, 64, 128, 256]},
        "LEARNING_RATE": {"values": [0.0001, 0.001]},
        "CROSS_ENTROPY_LOSS_WEIGHTS": {"values": [[1, 1, 1], [1, 1.5, 1.5], [1, 2, 2]]},
        "GRADIENT_CLIP": {"value": 1.0},
    },
}


def train_model(
    model: StockModel,
    wandb_run: wandb.Run,
    train_loader: DataLoader,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion: torch.nn.CrossEntropyLoss,
) -> StockModel:
    """
    Train a StockModel for direction classification.

    Args:
        model: The StockModel instance to train.
        wandb_run: Active wandb run for logging metrics and configuration.
        train_loader: DataLoader containing training batches.
        scaler: GradScaler for mixed precision training.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        criterion: CrossEntropyLoss for direction classification.

    Returns:
        StockModel: The trained model.
    """

    # get global vars
    global testX, testY

    # get the config
    run_config = wandb_run.config

    # early stopping variables
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None

    # get the actual preds
    actual_directions = testY[:, 0, 0].cpu().tolist()

    # train loop
    for epoch in range(run_config.EPOCHS):

        # switch model to training
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # get bacthes
        for x_batch, y_batch in train_loader:

            # reset the gradient
            optimizer.zero_grad(set_to_none=True)

            # use float16 or float32 based on the device
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):

                # model returns direction logits only
                direction_pred = model(x_batch)

                # get labels (label is now at index 0)
                labels = y_batch[:, 0, 0].long()

                # calculate loss
                loss = criterion(direction_pred, labels)

            # backward pass
            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.GRADIENT_CLIP)

            # take a step
            scaler.step(optimizer)
            scaler.update()

            # track metrics
            total_loss += loss.item()
            preds = direction_pred.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # calculate average loss
        avg_loss = total_loss / len(train_loader)

        # calculate train accuracy
        train_accuracy = correct / total

        # step scheduler
        scheduler.step(avg_loss)

        # wandb log
        wandb_run.log(
            {
                "avg_loss": avg_loss,
                "train_accuracy": train_accuracy,
            }
        )

        # evaluate the model
        direction_preds = evaluate_model(model, testX, run_config.BATCH_SIZE)

        # check how good the model was
        validation_accuracy = sum(
            p == a for p, a in zip(direction_preds, actual_directions)
        ) / len(actual_directions)

        # === EARLY STOPPING CHECK ===
        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # log the accuracy
        wandb_run.log(
            {
                "validation_accuracy": validation_accuracy,
            }
        )

        # log to console
        print(
            f"Epoch [{epoch + 1}/{run_config.EPOCHS}] Loss: {avg_loss:.4f} Acc: {train_accuracy:.4f} Val Acc: {validation_accuracy:.4f}"
        )

        # if patience is greater than tolerable
        if patience_counter >= run_config.VAL_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def evaluate_model(
    model: StockModel,
    testX: torch.Tensor,
    batch_size: int,
) -> List[int]:
    """
    Evaluate a StockModel on test data and generate direction predictions.

    Args:
        model: The StockModel instance to evaluate.
        testX: Test input tensor.
        batch_size: Batch size for inference.

    Returns:
        List[int]: Predicted direction labels.
    """

    # switch model for inference
    model.eval()
    direction_preds = []

    # with no gradient
    with torch.no_grad():

        # iterate
        for i in range(0, len(testX), batch_size):

            # get current batch
            x_batch = testX[i : i + batch_size]

            # model returns direction logits only
            direction_pred = model(x_batch)

            # get predicted class
            preds = direction_pred.argmax(dim=1).cpu().tolist()
            direction_preds.extend(preds)

    # return
    return direction_preds


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
    wandb.log(wandb.config)

    # setup training dataset
    train_dataset = TensorDataset(trainX, trainY)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sweep_config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # create the model for direction classification
    model = StockModel(
        input_size=sweep_config.INPUT_SIZE,
        lstm_layer_out=sweep_config.LSTM_HIDDEN_SIZE,
        lstm_layer_num=sweep_config.LSTM_NUM_LAYERS,
        lstm_dropout=sweep_config.LSTM_DROPOUT,
        fc1_out=sweep_config.FC1_OUT_FEATURES,
        fc_dropout=sweep_config.FC_DROPOUT,
        num_classes=sweep_config.NUM_CLASSES,
        device=DEVICE,
    )

    # create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=sweep_config.LEARNING_RATE)

    # create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )

    # create loss function with class weights
    class_weights = torch.tensor(
        sweep_config.CROSS_ENTROPY_LOSS_WEIGHTS, dtype=torch.float, device=DEVICE
    )
    criterion = NN.CrossEntropyLoss(weight=class_weights)

    # create a scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # train the model
    model = train_model(
        model=model,
        wandb_run=wandb_run,
        train_loader=train_loader,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )

    # evaluate the model
    direction_preds = evaluate_model(
        model=model,
        testX=testX,
        batch_size=sweep_config.BATCH_SIZE,
    )

    # calculate accuracy
    actual_directions = testY[:, 0, 0].cpu().tolist()
    correct = sum(p == a for p, a in zip(direction_preds, actual_directions))
    direction_accuracy = correct / len(actual_directions)

    # create confusion matrix
    cm_figure = create_confusion_matrix(actual_directions, direction_preds)

    # log to wandb
    wandb_run.log(
        {
            "direction_accuracy": direction_accuracy,
            "confusion_matrix": wandb.Image(cm_figure),
        }
    )

    # close figure to free memory
    plt.close(cm_figure)


if __name__ == "__main__":

    # login
    wandb.login()

    # constants
    CSV_FILE_NAME = "XRP_USDT.csv"
    FEATURE_COLS = [
        "open",
        "volume",
        "ema_0_diff",
        "ema_1_diff",
        "rsi_2",
        "adx_3",
        "return",
        "label",
    ]
    TARGET_COL = ["label"]

    # the csv file name
    dataframe_manager = DataManager(csv_file=CSV_FILE_NAME, device=DEVICE)

    # get training and testing tensors
    trainX, trainY = dataframe_manager.get_train_tensors(
        feature_col=FEATURE_COLS, target_col=TARGET_COL
    )
    testX, testY = dataframe_manager.get_test_tensors(
        feature_col=FEATURE_COLS, target_col=TARGET_COL
    )

    # create sweep
    # sweep_id = wandb.sweep(SWEPP_CONFIG, project="stock-prediction")
    # print("Sweep ID: ", sweep_id)

    # run sweep
    sweep_id = "w98fbu8r"
    wandb.agent(sweep_id=sweep_id, function=run_sweep, project="stock-prediction")
