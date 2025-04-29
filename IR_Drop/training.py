# For training the Unet model:
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from plotter import *
from tqdm import tqdm
from scipy.ndimage import zoom
from collections import defaultdict
from data_generation import Data_Generator
from torch.utils.data import Dataset, DataLoader

# Pre-Convolution Layer:
class PreConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=1),  # padding=1 to keep same size
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Dataset class working on npy files:
class IRDropDataset_Npy(Dataset):
    def __init__(self, feature_dir, label_dir, target_size=(256, 256)):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.feature_filenames = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npy')])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])

    def Resize(self, array, target_size):
        zoom_factors = [t / s for t, s in zip(target_size, array.shape)]
        return zoom(array, zoom_factors, order=1)

    def Normalize(self, arr):
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    def __len__(self):
        return len(self.feature_filenames)

    def __getitem__(self, idx):
        f = self.feature_filenames[idx]
        l = self.label_filenames[idx]
        features = np.load(os.path.join(self.feature_dir, f))
        label = np.load(os.path.join(self.label_dir, l))

        v = self.Normalize(features[1])
        c = self.Normalize(features[0])
        p = self.Normalize(features[2])
        y = self.Normalize(label)

        x = np.stack([c, v, p], axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return x, y

# Dataset class working on csv files:
class IRDropDataset_Csv(Dataset):
    def __init__(self, feature_dir, label_dir, target_size=(256, 256)):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.target_size = target_size

        # Build a list of testcases based on available files
        self.testcases = self.Build_Testcase_List()

    def Build_Testcase_List(self):
        testcase_set = set()
        for fname in os.listdir(self.feature_dir):
            if fname.startswith('current_map') and fname.endswith('.csv'):
                testcase = fname.replace('current_map_', '').replace('.csv', '')
                testcase_set.add(testcase)
        return sorted(list(testcase_set))

    def Resize(self, array, target_size):
        zoom_factors = [t / s for t, s in zip(target_size, array.shape)]
        return zoom(array, zoom_factors, order=1)

    def Normalize(self, arr):
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    def __len__(self):
        return len(self.testcases)

    def __getitem__(self, idx):
        testcase = self.testcases[idx]

        # Load each feature
        current_map = np.loadtxt(os.path.join(self.feature_dir, f'current_map_{testcase}.csv'), delimiter=',')
        voltage_map = np.loadtxt(os.path.join(self.feature_dir, f'voltage_source_map_{testcase}.csv'), delimiter=',')
        pdn_map = np.loadtxt(os.path.join(self.feature_dir, f'pdn_density_map_{testcase}.csv'), delimiter=',')

        # Load label
        label_path = os.path.join(self.label_dir, f'ir_drop_map_{testcase}.csv')
        label_map = np.loadtxt(label_path, delimiter=',')

        # Resize if needed
        current_map = self.Resize(current_map, self.target_size)
        voltage_map = self.Resize(voltage_map, self.target_size)
        pdn_map = self.Resize(pdn_map, self.target_size)
        label_map = self.Resize(label_map, self.target_size)

        # Normalize
        current_map = self.Normalize(current_map)
        voltage_map = self.Normalize(voltage_map)
        pdn_map = self.Normalize(pdn_map)
        label_map = self.Normalize(label_map)

        # Stack features
        x = np.stack([current_map, voltage_map, pdn_map], axis=0)  # Shape: (3, H, W)
        y = label_map[np.newaxis, :, :]                            # Shape: (1, H, W)

        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

# U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.preconv = PreConv(in_channels, in_channels)

        def Conv_Block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = Conv_Block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Conv_Block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottom = Conv_Block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_block2 = Conv_Block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block1 = Conv_Block(128, 64)
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
def Visualize_Prediction(x, y, pred, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    pred = pred.cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(y[0, 0], cmap='hot')
    axes[0].set_title("GT IR Drop")
    axes[1].imshow(pred[0, 0], cmap='hot')
    axes[1].set_title("Predicted IR Drop")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.close()

# Training loop with early stopping
def Train_Model(model, dataloader, device, epochs, lr, save_path, patience=15):
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

            if epoch % 2 == 0 and i == 0:
                Visualize_Prediction(x, y, pred, epoch, './Training_Predictions')

        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, f'unet_model.pth'))
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
    plt.savefig(os.path.join("./Training_Loss_Curve", 'training_loss_curve.png'))
    plt.show()

# Main entry point
def Train(model_path):
    feature_dir_npy = './Features'
    label_dir_npy = './Labels'
    feature_dir_csv = './Train_Data'
    label_dir_csv = './Train_Data'
    target_size = (256, 256)
    batch_size = 20
    epochs = 1
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = IRDropDataset_Npy(feature_dir_npy, label_dir_npy, target_size=target_size)
    # dataset = IRDropDataset_csv(feature_dir, label_dir, target_size=target_size)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = UNet(in_channels=3, out_channels=1)
    Train_Model(model, dataloader, device, epochs=epochs, lr=lr, save_path = model_path, patience=15)

# Generate Datapoint csvs:
def Generate(input_path, out_csv_path, mode):
    feature_path = "./Features_sample"
    label_path = "./Labels_sample"
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".sp"):
            Data_Generator(file_path, out_csv_path, feature_path, label_path, mode, gen_voltage_file=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-input", help = "Input: Path to CSV & Voltage files", type = str, required = True)
    parser.add_argument("-output", help = "Output: Path where the UNet model is stored", type = str, required = True)

    args = parser.parse_args()

    if args.input and args.output:
        input_dir = args.input
        ouput_dir = args.output
        os.makedirs(ouput_dir, exist_ok=True)
        
        print("######## Generating Training Data ########")
        out_csv_path = "./Train_Data_sample"
        mode = 'train'
        Generate(input_dir, out_csv_path, mode)

        print("######## Training Model ########")
        Train(ouput_dir)