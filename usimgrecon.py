import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
!pip install torchmetrics
import scipy.io
from torch.utils.data import Dataset
from scipy.interpolate import interp1d # For delay adjustment in ultrasound using 1D cubic interpolation
from scipy.ndimage import zoom
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import zipfile
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

from google.colab import drive
drive.mount('/content/drive')

# Step 1: Download the zip file from Google Drive
file_id = '   '  # Replace with your file ID
output_path = '  ' # replace with output path
#gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)# Path to your zip file in Google Drive
zip_path = ' '

# Directory where you want to extract the files
extract_dir = '/content'

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Check extracted files
os.listdir(extract_dir)class CustomDataset(Dataset):  # Renamed to follow CamelCase

    @staticmethod
    def getFiles(path):
        paths = []
        for root, _, files in os.walk(path):
            for file in files:
                paths.append(os.path.join(root, file))
        return paths

    @staticmethod
    def readFile(path):
        with h5py.File(path, "r") as dataFrame:
            inp = np.array(dataFrame['inp'], dtype="float32")  / 16384
            out = np.array(dataFrame['out'], dtype="float32")

        return inp, out

    def __init__(self, path):
        super(CustomDataset, self).__init__()
        self.filePaths = self.getFiles(path)

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, index):
        if index < len(self.filePaths):
            path = self.filePaths[index]
            inp, out = self.readFile(path)

            inp = torch.from_numpy(inp).float()  # Use torch.from_numpy for better performance

            out = torch.from_numpy(out).float()

            return {
                'input': inp,
                'output': out
            }
class AntiRectifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)  # Subtract mean along dimension 1
        x = F.normalize(x, p=2, dim=1)  # Normalize across dimension 1
        pos_neg = torch.cat([F.relu(x), F.relu(-x)], dim=1)  # Combine positive and negative rectified parts
        return pos_neg


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = AntiRectifier()

        self.conv2 = nn.Conv2d(2 * mid_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.act2 = AntiRectifier()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Interpolate x1 to match the size of x2 instead of using manual padding
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate and perform convolution
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNetBeamformer(nn.Module):
    def __init__(self, n_channels=128, bilinear=False):
        super(UNetBeamformer, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256 , bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return F.softmax(x, dim=1)

############# MobileNet ##################

class AntiRectifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1)
        pos_neg = torch.cat([F.relu(x), F.relu(-x)], dim=1)
        return pos_neg

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = AntiRectifier()

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels , out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = AntiRectifier()

    def forward(self, x):
        x = F.elu(self.depthwise(x))
        x = F.elu(self.pointwise(x))
        x = self.bn(x)
        x = self.act(x)
        return x




class MobileNet(nn.Module):
    def __init__(self,n_channels=128):
        super(MobileNet, self).__init__()

        #First conv layer
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = AntiRectifier()

        #Depthwise Separable Convolutional Layers
        self.dw_conv2 = DepthwiseSeparableConv(64, 64)  # Input channels doubled after act1
        self.dw_conv3 = DepthwiseSeparableConv(128, 128)
        self.dw_conv4 = DepthwiseSeparableConv(256, 256)
        self.dw_conv5 = DepthwiseSeparableConv(512, 512)

        # After final depthwise separable conv, reduce channels back to 128
        self.channel_reduce = nn.Conv2d(512, 128, kernel_size=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)  # Channels doubled to 64

        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        x = self.dw_conv4(x)

        x = self.channel_reduce(x)  # Reduces final output channels to 128

        return F.softmax(x, dim=1)

### Simple CNN #######

import torch
import torch.nn as nn
import torch.nn.functional as F

class AntiRectifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1)
        pos_neg = torch.cat([F.relu(x), F.relu(-x)], dim=1)
        return pos_neg

class CNN(nn.Module):
    def __init__(self,n_channels=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = AntiRectifier()

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # Adjusted to maintain output at 128 channels
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = AntiRectifier()

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = AntiRectifier()

        #  layer at the end to ensure the output has 128 channels
        self.channel_reduce = nn.Conv2d(512, 128, kernel_size=1)


        self.output_channels = 128

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)  # Doubles the channels from 64 to 128

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x= self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        # Reduce channels to 128
        x = self.channel_reduce(x)

        return F.softmax(x, dim=1)


#Training
class Trainer():

    @staticmethod
    def norm(x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    def __init__(self, dataset, args, loss='MSE', split=0.8):
        self.bs = args.bs
        self.lr = args.lr
        # self.model = UNetBeamformer()
        self.model = MobileNet()
        self.model = CNN()

        self.criterion = L1Loss() if (loss == 'MAE') else MSELoss()

        # Train/Validation split
        self.train_size = int(len(dataset) * split)
        self.valid_size = len(dataset) - self.train_size
        self.train_set, self.valid_set = random_split(
            dataset, [self.train_size, self.valid_size], generator=torch.Generator().manual_seed(42)
        )

        # DataLoader with configurable number of workers
        self.train_data = DataLoader(self.train_set, batch_size=self.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        self.valid_data = DataLoader(self.valid_set, batch_size=self.bs, num_workers=args.num_workers, pin_memory=True)

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.save_paths = args.save

    def train(self, epochs, run_no):
        # Set up directories and logging
        save_paths = self.save_paths
        os.makedirs(os.path.join(save_paths, "chkpt/", 'iter_' + str(run_no)), exist_ok=True)
        os.makedirs(os.path.join(save_paths, "logs/"), exist_ok=True)
        writer = SummaryWriter(os.path.join(save_paths, 'logs/iter_' + str(run_no)))

        chkpt = os.path.join(save_paths, "chkpt", 'iter_' + str(run_no), "model.pt")
        best = os.path.join(save_paths, "chkpt", 'iter_' + str(run_no), "best.pt")
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scaler = GradScaler()
        #scheduler = StepLR(optimizer, step_size=100, gamma=0.5)  # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.1, verbose=True)


        # Load checkpoint if exists
        if os.path.exists(chkpt):
            checkpoint = torch.load(chkpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
            start = checkpoint['epoch']
            train_step = checkpoint['train_step']
            val_step = checkpoint['val_step']
            threshold = checkpoint['loss']
        else:
            start = 0
            train_step = 0
            val_step = 0
            threshold = 1000
            ground_truth_logged = True

        # Training loop
        for epoch in range(start, epochs):
            self.model.train()
            train_loss = 0
            val_loss = 0

            for i, batch in enumerate(self.train_data):
                input_data = batch['input'].to(self.device)
                output = batch['output'].to(self.device)
                # print("Input_train: ", input_data.shape)
                # print("output_train: ", output.shape)

                with autocast():
                    pred = self.model(input_data)
                    # pred = pred.unsqueeze(1)
                    # print("Pred_train: ", pred.shape)
                    beamformed = torch.mul(pred, input_data)
                    beamformed_sum = torch.sum(beamformed, axis=1)
                    beamformed_sum = HilbertTransform(axis=1)(beamformed_sum)
                    envelope = torch.abs(beamformed_sum)
                    imPred = 20 * torch.log10(envelope / torch.clip(torch.max(envelope), min=1e-8))
                    loss = self.criterion(imPred, output) / 4  # Gradient accumulation over 4 steps

                # loss.backward()  # Backpropagation
                # optimizer.step()  # Update the weights

                scaler.scale(loss).backward()

                # Accumulate gradients and update every 4 steps
                if (i + 1) % 4 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * 4  # Undo the division for logging purposes

                train_step += 1

            train_loss /= len(self.train_set)

            # Validation loop
            val_loss, ground_truth_logged = self.validate(val_step, writer, epoch, ground_truth_logged)

            print(f'Epoch = {epoch:3d}, Training Loss = {train_loss:.3f}, Validation Loss = {val_loss:.3f}')
            writer.add_scalar('Train Loss', train_loss, global_step=epoch)
            writer.add_scalar('Validation Loss', val_loss, global_step=epoch)
            # Learning rate scheduler step
            scheduler.step(val_loss)

            # Save checkpoint
            if epoch % 5 == 0 or val_loss < threshold:
                print("Saving model checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    'train_step': train_step,
                    'val_step': val_step,
                }, chkpt)

            # Save the best model
            if val_loss < threshold:
                print("Saving best model weights")
                threshold = val_loss
                torch.save(self.model.state_dict(), best)

            writer.flush()

        writer.close()

    def validate(self, val_step, writer, epoch, ground_truth_logged):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in self.valid_data:
                input_data = batch['input'].to(self.device)
                output = batch['output'].to(self.device)
                # print("Input_val: ", input_data.shape)
                # print("output_val: ", output.shape)

                with autocast():
                    pred = self.model(input_data)
                    # pred = pred.unsqueeze(1)
                    # print("Pred_val: ", pred.shape)
                    beamformed = torch.mul(pred, input_data)
                    beamformed_sum = torch.sum(beamformed, axis=1)
                    beamformed_sum = HilbertTransform(axis=1)(beamformed_sum)
                    envelope = torch.abs(beamformed_sum)
                    imPred = 20 * torch.log10(envelope / torch.clip(torch.max(envelope), min=1e-8))
                    loss = self.criterion(imPred, output)

                val_loss += loss.item()
                val_step += 1

        val_loss /= len(self.valid_set)

        # Now save the best image to see the iteration

        # Convert to numpy for visualization
        imPred = self.norm(imPred)
        output = self.norm(output)
        imPred = imPred.detach().cpu().numpy()
        output = output.detach().cpu().numpy()


        # TensorBoard Image logging
        # Log ground truth only once, during the first batch
        if ground_truth_logged:
            writer.add_image('GT', output[0], global_step=epoch, dataformats='HW')
        writer.add_image('Pred'+str(epoch), imPred[0], global_step=epoch, dataformats='HW')

        # Visualization using matplotlib
        # Display the ground truth and prediction for visual reference, only once
        if ground_truth_logged:
            plt.imshow(output[0], cmap='gray', aspect='auto')
            plt.axis('off')
            plt.title('Ground Truth')
            plt.show()
            ground_truth_logged = False  # Ensure ground truth is logged once

        plt.imshow(imPred[0], cmap='gray', aspect='auto')
        plt.axis('off')

        return val_loss,ground_truth_logged

DATASET = CustomDataset(args.data)

T = Trainer(DATASET, args)
T.train(epochs=30, run_no=args.run)

#Testing 
ground_truth_path = "" # path to the gorund truth image 
test_data_path = "" #path to the testing data
ground_truth = scipy.io.loadmat(ground_truth_path)
test_data = scipy.io.loadmat(test_data_path)
ground_truth_data = ground_truth['groundtruth_image']  
test_data_data = test_data['resampled_tof_corrected_data'] 
ground_truth_tensor = torch.tensor(ground_truth_data, dtype=torch.float32)  
test_data_tensor = torch.tensor(test_data_data, dtype=torch.float32).unsqueeze(0)  
#model_path = "" #path to youe model
  
 
#model = UNetBeamformer() 
#model = MobileNet() 
model = CNN() 
# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
model.eval() 


with torch.no_grad():
    model_output = model(test_data_tensor) 
    beamformed = torch.mul(model_output, test_data_tensor)  
    beamformed_sum = torch.sum(beamformed, dim=1)  
    envelope = torch.abs(beamformed_sum.squeeze(0)) 
    imPred = 20 * torch.log10(envelope / torch.clamp(torch.max(envelope), min=1e-8))  

mse_loss_fn = MSELoss()
mae_loss_fn = L1Loss()
ssim_metric = SSIM(data_range=1.0)  
imPred_normalized = (imPred - imPred.min()) / (imPred.max() - imPred.min())
ground_truth_normalized = (ground_truth_tensor - ground_truth_tensor.min()) / (
    ground_truth_tensor.max() - ground_truth_tensor.min()
)
mse_loss = mse_loss_fn(imPred, ground_truth_tensor).item()
mae_loss = mae_loss_fn(imPred, ground_truth_tensor).item()
ssim_score = ssim_metric(
    imPred_normalized.unsqueeze(0).unsqueeze(0),
    ground_truth_normalized.unsqueeze(0).unsqueeze(0),
).item()

print("MSE loss , MAE loss , SSIM score", mse_loss,mae_loss,ssim_score)

plt.figure(figsize=(10, 5))
plt.imshow(ground_truth_tensor.cpu().numpy(), cmap='gray')
plt.title("Ground Truth")
plt.axis('off')
plt.savefig('ground_truth_image.png')  
plt.show()
plt.figure(figsize=(10, 5))
plt.imshow(imPred.cpu().numpy(), cmap='gray')
plt.title("Beamformed Prediction")
plt.axis('off')
plt.savefig('beamformed_prediction.png')  
plt.show()




