"""
Plot prediction results: confusion matrix for direction and line graphs for max/min predictions.
"""

# 3rd party imports
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# local imports
from train import evaluate_model
from settings import project_settings
from dataset_handler import DataManager

# --------------------------------------------------
# GLOBAL SETTINGS
# --------------------------------------------------
DEVICE = project_settings.DEVICE


# --------------------------------------------------
# HYPERPARAMETERS
# --------------------------------------------------
BATCH_SIZE = 64
MODEL_PATH = "model.pt"


if __name__ == "__main__":

    # load the data
    csv_file_name = "XRP_USDT.csv"
    dataframe_manager = DataManager(
        csv_file=csv_file_name,
        device=DEVICE,
    )

    # the columns to use
    feature_cols = ["open", "EMA_0_diff", "EMA_1_diff", "ADX_2", "RSI_3"]
    target_col = ["future_max", "future_min", "label"]

    # get testing tensors
    testX, testY = dataframe_manager.get_test_tensors(
        feature_col=feature_cols, target_col=target_col
    )

    # setup training dataset
    train_dataset = TensorDataset(testX, testY)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # load the model
    model = torch.jit.load(MODEL_PATH)

    # --------------------------------------------------
    # EVALUATE MODEL
    # --------------------------------------------------
    max_preds, min_preds, direction_preds = evaluate_model(model, testX, BATCH_SIZE)

    # bring to cpu for plotting
    actual_maxs = testY[:, 0, 0].cpu().numpy()
    actual_mins = testY[:, 1, 0].cpu().numpy()
    actual_directions = testY[:, 2, 0].cpu().numpy().astype(int)

    # --------------------------------------------------
    # PLOT CONFUSION MATRIX FOR DIRECTION PREDICTIONS
    # --------------------------------------------------
    cm = confusion_matrix(actual_directions, direction_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Neutral", "Up", "Down"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Direction Prediction Confusion Matrix")
    plt.tight_layout()
    plt.savefig("pred_res/direction_confusion_matrix.png", dpi=150)
    plt.show()

    # --------------------------------------------------
    # PLOT LINE GRAPH FOR MAX PREDICTIONS
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_maxs, label="Actual Max", alpha=0.7)
    ax.plot(max_preds, label="Predicted Max", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Max Price")
    ax.set_title("Actual vs Predicted Maximum Price")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pred_res/max_prediction_line_graph.png", dpi=150)
    plt.show()

    # --------------------------------------------------
    # PLOT LINE GRAPH FOR MIN PREDICTIONS
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual_mins, label="Actual Min", alpha=0.7)
    ax.plot(min_preds, label="Predicted Min", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Min Price")
    ax.set_title("Actual vs Predicted Minimum Price")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pred_res/min_prediction_line_graph.png", dpi=150)
    plt.show()

    print("Plots saved to pred_res/ directory")
