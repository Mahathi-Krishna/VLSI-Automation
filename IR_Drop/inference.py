import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from training import UNet
from scipy.ndimage import zoom
from data_generation import Data_Generator

# -----------------------------
# Helper Functions
# -----------------------------
def resize(array, target_size=(256, 256)):
    zoom_factors = [t / s for t, s in zip(target_size, array.shape)]
    return zoom(array, zoom_factors, order=1)

def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

def plot_comparison(y_true, y_pred, filename, save_dir):
    y_true = y_true.squeeze().cpu().numpy()
    y_pred = y_pred.squeeze().cpu().numpy()
    abs_error = np.abs(y_true - y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    im0 = axes[0].imshow(y_true, cmap='jet')
    axes[0].set_title('Ground Truth IR Drop')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, cmap='jet')
    axes[1].set_title('Predicted IR Drop')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{filename}.png'))
    plt.close()

def compute_metrics(preds, targets):
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    mse_val = mse_loss(preds, targets).item()
    l1_val = l1_loss(preds, targets).item()
    mae_val = torch.abs(preds - targets).mean().item()

    max_val = targets.max()
    threshold = 0.9 * max_val

    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()

    TP = (preds_bin * targets_bin).sum()
    FP = (preds_bin * (1 - targets_bin)).sum()
    FN = ((1 - preds_bin) * targets_bin).sum()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return mse_val, l1_val, mae_val, f1.item(), precision.item(), recall.item()

def save_prediction_as_csv(pred, output_dir, filename_base):
    pred_np = pred.squeeze().cpu().numpy()
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{filename_base}_predicted.csv")
    np.savetxt(csv_path, pred_np, delimiter=',')
    print(f"Saved prediction to {csv_path}")

# -----------------------------
# Main Inference Function
# -----------------------------
def model_inference_npy(model_path, feature_dir, label_dir, filename, pred_save_dir, metrics_save_dir, target_size=(256,256)):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    features = np.load(os.path.join(feature_dir, f"feature_{filename}.npy"))
    label = np.load(os.path.join(label_dir, f"label_{filename}.npy"))

    # Resize to target size
    current_map = normalize(features[0])
    voltage_map = normalize(features[1])
    pdn_map = normalize(features[2])
    label_map = normalize(label)

    # Stack features
    x = np.stack([current_map, voltage_map, pdn_map], axis=0)

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(label_map, dtype=torch.float32).unsqueeze(0)

    # Load model
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # Metrics
        mse_val, l1_val, mae_val, f1_val, prec_val, rec_val = compute_metrics(pred, y)

        # Plot comparison
        plot_comparison(y, pred, filename, pred_save_dir)

        # Save prediction
        save_prediction_as_csv(pred, pred_save_dir, filename)

        # Save metrics
        metrics = {
            'Sample': filename,
            'MSE': mse_val,
            'L1 Loss': l1_val,
            'MAE': mae_val,
            'F1 Score': f1_val,
            'Precision': prec_val,
            'Recall': rec_val
        }
        metrics_df = pd.DataFrame([metrics])
        os.makedirs(metrics_save_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(metrics_save_dir, f'{filename}_metrics.csv'), index=False)

        print(f"Metrics saved to {metrics_save_dir}")


def model_inference_csv(model_path, feature_dir, label_dir, filename, pred_save_dir, metrics_save_dir, target_size=(256,256)):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare paths
    current_map_path = os.path.join(feature_dir, f'current_map_{filename}.csv')
    voltage_map_path = os.path.join(feature_dir, f'voltage_source_map_{filename}.csv')
    pdn_map_path = os.path.join(feature_dir, f'pdn_density_map_{filename}.csv')
    ir_drop_path = os.path.join(label_dir, f'ir_drop_map_{filename}.csv')

    # Load data
    current_map = np.loadtxt(current_map_path, delimiter=',')
    voltage_map = np.loadtxt(voltage_map_path, delimiter=',')
    pdn_map = np.loadtxt(pdn_map_path, delimiter=',')
    label_map = np.loadtxt(ir_drop_path, delimiter=',')

    # Resize to target size
    current_map = resize(current_map, target_size)
    voltage_map = resize(voltage_map, target_size)
    pdn_map = resize(pdn_map, target_size)
    label_map = resize(label_map, target_size)

    # Normalize
    current_map = normalize(current_map)
    voltage_map = normalize(voltage_map)
    pdn_map = normalize(pdn_map)
    label_map = normalize(label_map)

    # Stack features
    x = np.stack([current_map, voltage_map, pdn_map], axis=0)
    y = label_map[np.newaxis, :, :]

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 3, H, W)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, 1, H, W)

    # Load model
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # Metrics
        mse_val, l1_val, mae_val, f1_val, prec_val, rec_val = compute_metrics(pred, y)

        # Plot comparison
        plot_comparison(y, pred, filename, pred_save_dir)

        # Save prediction
        save_prediction_as_csv(pred, pred_save_dir, filename)

        # Save metrics
        metrics = {
            'Sample': filename,
            'MSE': mse_val,
            'L1 Loss': l1_val,
            'MAE': mae_val,
            'F1 Score': f1_val,
            'Precision': prec_val,
            'Recall': rec_val
        }
        metrics_df = pd.DataFrame([metrics])
        os.makedirs(metrics_save_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(metrics_save_dir, f'{filename}_metrics.csv'), index=False)

        print(f"Metrics saved to {metrics_save_dir}")

# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-spice_file", help="Input: Path to Spice Netlist", type=str, required=True)
    parser.add_argument("-ml_model", help="Input: Path to saved ML model", type=str, required=True)
    parser.add_argument("-output", help="Output: Path where the prediction CSV and plots are saved", type=str, required=True)

    args = parser.parse_args()

    spice_file = args.spice_file
    model_path = args.ml_model
    output_path = args.output

    os.makedirs(output_path, exist_ok=True)

    # Prepare filename
    filename = os.path.basename(spice_file)
    filename = os.path.splitext(filename)[0]

    # Step 1: Generate CSVs from Netlist
    mode = 'test'
    test_csv_path = "./Test_Data"
    test_feature_path = os.path.join(test_csv_path, "Features")
    test_label_path = os.path.join(test_csv_path, "Labels")
    
    os.makedirs(test_csv_path, exist_ok=True)
    os.makedirs(test_feature_path, exist_ok=True)
    os.makedirs(test_label_path, exist_ok=True)

    Data_Generator(spice_file, test_csv_path, test_feature_path, test_label_path, mode, gen_voltage_file=True)

    # Step 2: Run Inference
    
    # Inference using csv files:
    # model_inference_csv(
    #     model_path = model_path,
    #     feature_dir = test_csv_path,
    #     label_dir = test_csv_path,
    #     filename = filename,
    #     pred_save_dir = output_path,
    #     metrics_save_dir = output_path,
    #     target_size = (256, 256)
    # )

    # Inference using npy files:
    model_inference_npy(
        model_path = model_path,
        feature_dir = test_feature_path,
        label_dir = test_label_path,
        filename = filename,
        pred_save_dir = output_path,
        metrics_save_dir = output_path,
        target_size = (256, 256)
    )