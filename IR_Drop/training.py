import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

class PreConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=1),  # padding=1 to keep same size
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Dataset class
class IRDropDataset(Dataset):
    def __init__(self, feature_dir, label_dir, target_size=(256, 256)):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.filenames = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npy')])

    def resize(self, array, target_size):
        zoom_factors = [t / s for t, s in zip(target_size, array.shape)]
        return zoom(array, zoom_factors, order=1)

    def normalize(self, arr):
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = self.filenames[idx]
        features = np.load(os.path.join(self.feature_dir, f))
        label = np.load(os.path.join(self.label_dir, f))

        # v = self.normalize(self.resize(features['current_map'], self.target_size))
        # c = self.normalize(self.resize(features['effective_voltage_dist_map'], self.target_size))
        # p = self.normalize(self.resize(features['pdn_map'], self.target_size))
        # y = self.resize(label['ir_drop_map'], self.target_size)

        v = self.normalize(features[1])
        c = self.normalize(features[0])
        p = self.normalize(features[2])
        y = self.normalize(label)

        x = np.stack([v, c, p], axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return x, y

# U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.preconv = PreConv(in_channels, in_channels)

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottom = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_block2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        b = self.bottom(p2)
        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_block2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_block1(u1)
        return self.final(u1)

# Visualization for debugging
def visualize_prediction(x, y, pred, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    pred = pred.cpu().detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x[1, 1], cmap='viridis')
    axes[0].set_title("Current Map")
    axes[1].imshow(y[0, 0], cmap='hot')
    axes[1].set_title("GT IR Drop")
    axes[2].imshow(pred[0, 0], cmap='hot')
    axes[2].set_title("Predicted IR Drop")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.close()

# Training loop with early stopping
def train_model(model, dataloader, device, epochs=300, lr=1e-4, save_path='./models', patience=15):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    os.makedirs(save_path, exist_ok=True)

    best_loss = float('inf')
    patience_counter = 0
    loss_list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (x, y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = mse(pred, y) + 0.1 * l1(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch % 5 == 0 and i == 0:
                visualize_prediction(x, y, pred, epoch, './vis_debugzz')

        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, f'unet_without_filter.pth'))
            print("Model improved and saved.")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_list)+1), loss_list, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join("./Loss_Curve", 'training_loss_curve.png'))
    plt.show()

# Main entry point
def main():
    feature_dir = './Without_Filter/Features'
    label_dir = './Without_Filter/Labels'
    target_size = (256, 256)
    batch_size = 20
    epochs = 300
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = IRDropDataset(feature_dir, label_dir, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = UNet(in_channels=3, out_channels=1)
    train_model(model, dataloader, device, epochs=epochs, lr=lr, save_path='./models', patience=15)

if __name__ == "__main__":
    main()
