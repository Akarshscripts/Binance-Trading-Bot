"""
This file is used to train the model using hyperparameter sweep and wandb.
"""

# 3rd party imports
import wandb
import torch
import numpy as np
from torch import nn as NN
from torch.utils.data import DataLoader, TensorDataset

# local imports
from settings import Constants
from neural_net import StockModel
from dataframe_handler import DataManager
from indicators import RSI, BollingerBands, VWAP, ATR

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
    "name": "new_model_sweep_1",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "mae_accuracy"},
    "parameters": {
        "BATCH_SIZE": {"values": [32, 64, 128, 256]},
        "FC1_INPUT_SIZE": {"values": [32, 64, 128]},
        "FC2_INPUT_SIZE": {"values": [16, 32, 64]},
        "DROPOUT": {"values": [0.2, 0.3, 0.4]},
        "EPOCHS": {"value": 192},
    },
}


def train_model(
    model: StockModel,
    wandb_run: wandb.Run,
    test_loader: DataLoader,
    train_loader: DataLoader,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion: torch.nn.HuberLoss,
) -> StockModel:
    """
    Train a StockModel for stock price prediction.

    Args:
        model: The StockModel instance to train.
        wandb_run: Active wandb run for logging metrics and configuration.
        test_loader: DataLoader containing test batches for evaluation.
        train_loader: DataLoader containing training batches.
        scaler: GradScaler for mixed precision training.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        criterion: Loss function for training.

    Returns:
        StockModel: The trained model.
    """

    # get the config
    run_config = wandb_run.config

    # early stopping variables
    best_val_mae = float("inf")
    patience_counter = 0
    best_model_state = None

    # valid steps
    valid_steps = 0

    # train loop
    for epoch in range(run_config.EPOCHS):

        # switch model to training
        model.train()
        total_loss = 0.0
        avg_loss = 0.0

        # get bacthes
        for x_batch, y_batch in train_loader:

            # reset the gradient
            optimizer.zero_grad(set_to_none=True)

            # use float16 or float32 based on the device
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):

                # predict and calc loss
                preds = model(x_batch)
                loss = criterion(preds, y_batch)

                # check if loss is finite
                if not torch.isfinite(loss):
                    continue

            # backward pass
            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Constants.GRADIENT_CLIP)

            # take a step
            scaler.step(optimizer)
            scaler.update()

            # track metrics
            total_loss += loss.item()
            valid_steps += 1

        # calculate average loss
        avg_loss = total_loss / max(1, valid_steps)

        # wandb log
        wandb_run.log({"avg_loss": avg_loss})

        # evaluate the model
        mae = evaluate_model(model, test_loader)

        # step the scheduler
        scheduler.step(mae)

        # === EARLY STOPPING CHECK ===
        if mae < best_val_mae:
            best_val_mae = mae
            patience_counter = 0
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        # log the accuracy
        wandb_run.log({"mae_accuracy": mae})

        # log to console
        print(
            f"Epoch [{epoch + 1}/{run_config.EPOCHS}] Loss: {avg_loss:.4f} MAE: {mae:.4f}"
        )

        # if patience is greater than tolerable
        if patience_counter >= Constants.VAL_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def evaluate_model(
    model: StockModel,
    test_loader: DataLoader,
) -> float:
    """
    Evaluate a StockModel on test data and generate direction predictions.

    Args:
        model: The StockModel instance to evaluate.
        testX: Test input tensor.
        batch_size: Batch size for inference.

    Returns:
        float: The mean absolute error of the predictions.
    """

    # switch model for inference
    model.eval()
    r_preds = []
    actual_r_multiple_values = []

    # with no gradient
    with torch.no_grad():

        # iterate
        for x_batch, y_batch in test_loader:

            # model returns the predicted r
            r_pred = model(x_batch)
            r_preds.append(r_pred.cpu())
            actual_r_multiple_values.append(y_batch.cpu())

    r_preds = torch.cat(r_preds).numpy()
    actual_r_multiple_values = torch.cat(actual_r_multiple_values).numpy()
    mae = np.mean(np.abs(r_preds - actual_r_multiple_values))

    # return
    return mae


def run_sweep():
    """
    Run a hyperparameter sweep experiment using wandb.
    This function is called by wandb.agent to train and evaluate a model
    with hyperparameters specified in the sweep configuration.
    """

    global trainX, trainY, testX, testY

    # init wandp
    wandb_run = wandb.init(reinit="finish_previous")
    sweep_config = wandb_run.config

    # log the config once
    wandb.log(sweep_config.as_dict())

    # setup training dataset
    train_dataset = TensorDataset(trainX, trainY)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sweep_config.BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )

    # setup testing dataset
    test_dataset = TensorDataset(testX, testY)
    test_loader = DataLoader(
        test_dataset,
        batch_size=sweep_config.BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )

    # create the model for direction classification
    model = StockModel(
        fc1_input_size=sweep_config.FC1_INPUT_SIZE,
        fc2_input_size=sweep_config.FC2_INPUT_SIZE,
        dropout=sweep_config.DROPOUT,
        device=DEVICE,
    )

    # create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Constants.LEARNING_RATE)

    # create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )

    # create loss function
    criterion = NN.HuberLoss()

    # create a scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # train the model
    model = train_model(
        model=model,
        wandb_run=wandb_run,
        test_loader=test_loader,
        train_loader=train_loader,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )

    # evaluate the model
    final_mae = evaluate_model(
        model=model,
        test_loader=test_loader,
    )

    # log to wandb
    wandb_run.log({"final_mae_accuracy": final_mae})


if __name__ == "__main__":

    # login
    wandb.login()

    # create indicators
    bband = BollingerBands(20)
    vwap = VWAP()
    atr = ATR(14)
    rsi = RSI(14)
    indicators = [bband, vwap, atr, rsi]

    # create datafram manager
    dm = DataManager(
        device=DEVICE,
        split_ratio=Constants.TRAIN_RATIO,
        csv_file=Constants.PROCESSED_FILE_PATH,
    )

    # # implement preprocessing
    # dm.compute_indicators(
    #     indicators=indicators,
    # )

    # # add r-multiple-cols
    # dm.add_r_multiple(
    #     trade_length=Constants.MAX_ALLOWED_TRADE_LENGTH,
    #     atr_col_name=Constants.ATR_COL_NAME,
    #     reward_r=Constants.R_MULTIPLE_REWARD,
    # )

    # # scale columns
    # dm.scale_cols(cols=Constants.SCALABLE_COLS)

    # # stack features
    # dm.stack_features(
    #     feature_cols=Constants.COLS_TO_STACK,
    #     lags=sweep_config.COL_LAGS,
    # )

    # # drop columns
    # dm.df.drop(columns=Constants.COLS_TO_DROP, inplace=True)

    # get train/test data
    trainX, trainY = dm.get_train_tensors(Constants.TARGET_COL)
    testX, testY = dm.get_test_tensors(Constants.TARGET_COL)

    # create sweep
    # sweep_id = wandb.sweep(SWEPP_CONFIG, project="stock-prediction")
    # print("Sweep ID: ", sweep_id)

    # run sweep
    sweep_id = "ix4v72ds"
    wandb.agent(sweep_id=sweep_id, function=run_sweep, project="stock-prediction")
