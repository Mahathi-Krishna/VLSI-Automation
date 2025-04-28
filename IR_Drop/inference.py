import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from training import IRDropDataset, UNet  # Import your existing Dataset and Model

# -----------------------------
# Helper Functions
# -----------------------------

def plot_comparison(x, y_true, y_pred, idx, save_dir='./comparison_plots'):
    os.makedirs(save_dir, exist_ok=True)

    y_true = y_true.squeeze().cpu().numpy()
    y_pred = y_pred.squeeze().cpu().numpy()
    abs_error = np.abs(y_true - y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(y_true, cmap='jet')
    axes[0].set_title('Ground Truth IR Drop')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, cmap='jet')
    axes[1].set_title('Predicted IR Drop')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(abs_error, cmap='hot')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_{idx}_comparison.png'))
    plt.close()

    print(f"Saved comparison plot for sample {idx}")

def compute_normalized_mae(preds, targets):
    abs_diff = torch.abs(preds - targets)
    mae = abs_diff.mean().item()
    return mae

def compute_f1_score_custom(preds, targets):
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

    return f1.item(), precision.item(), recall.item()

def save_prediction_as_csv(pred, output_dir, filename_base):
    os.makedirs(output_dir, exist_ok=True)
    pred_np = pred.squeeze().cpu().numpy()
    csv_path = os.path.join(output_dir, f"{filename_base}_predicted.csv")
    np.savetxt(csv_path, pred_np, delimiter=',')
    print(f"Saved prediction to {csv_path}")

# -----------------------------
# Test Function
# -----------------------------
def test_and_save_results(model_path, feature_dir, label_dir,
    pred_save_dir='./predicted_csv',
    metrics_save_path='./test_metrics_summary.csv',
    target_size=(256, 256), batch_size=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = IRDropDataset(feature_dir, label_dir, target_size=target_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    metrics_list = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Loss Metrics
            mse_val = mse_loss(pred, y).item()
            l1_val = l1_loss(pred, y).item()
            mae_val = compute_normalized_mae(pred, y)
            f1_val, prec_val, rec_val = compute_f1_score_custom(pred, y)

            plot_comparison(x, y, pred, idx, save_dir='./comparison_plots')

            # Save prediction as CSV
            filename_base = f"sample_{idx}"
            save_prediction_as_csv(pred, pred_save_dir, filename_base)

            # Store metrics
            metrics_list.append({
                'Sample': filename_base,
                'MAE': mae_val,
                'MSE': mse_val,
                'L1 Loss': l1_val,
                'F1 Score': f1_val,
                'Precision': prec_val,
                'Recall': rec_val
            })

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"\n✅ Saved all test metrics to {metrics_save_path}")

# -----------------------------
# Main Entry
# -----------------------------
if __name__ == "__main__":
    test_and_save_results(
        model_path='./models/unet_preconv.pth',          # Your model path
        feature_dir='./Predict/Features',                # Your feature .npy files
        label_dir='./Predict/Labels',                    # Your label .npy files
        pred_save_dir='./Predicted_CSV',                 # Where to save predicted maps
        metrics_save_path='./Test_Metrics_Summary.csv',  # Where to save metrics
        target_size=(256, 256),
        batch_size=1
    )