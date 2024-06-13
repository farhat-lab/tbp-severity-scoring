from glob import glob
from datetime import date, datetime
from sklearn import metrics
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import json
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch  
import torchvision
import torchxrayvision as xrv
import pandas as pd
import pylab
import os
import sklearn
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import torchxrayvision as xrv
import wandb


# Specify Project details

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, '../data')
IMAGE_METADATA_PATH = os.path.join(DATA_PATH, "tbp_cxr_metadata.csv")  # sample CSV file containing labels

class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the image path and label from the DataFrame
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['normalized_log_overallpercentofabnormalvolume'] # Adjust 'label' to the name of your actual label column
    
        # Load the image
        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)  # Normalize the image
        img = img[None, ...]  # Add a channel dimension

        # Apply transformations if any
        if self.transform:
            img = self.transform(img)

        # Convert img to tensor
        img = torch.from_numpy(img).float()
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float)
        # Optionally, return the image ID as well for tracking
        image_id = self.dataframe.iloc[idx]['patient_id']

        # Return a dictionary of the data
        return {'img': img, 'label': label, 'image_id': image_id}
    

def inverse_min_max_scaling(norm_value, log_min, log_max):
    return norm_value * (log_max - log_min) + log_min

def inverse_log_transformation(log_value):
    return np.exp(log_value) - 1

    
def get_training_image_path(number):
    filename = f"{number}.png"
    path = DATA_PATH + 'train/'
    filename = path+filename
    return filename

main_df = pd.read_csv(IMAGE_METADATA_PATH)
main_df['log_overallpercentofabnormalvolume'] = np.log1p(main_df['overallpercentofabnormalvolume'])
log_min = main_df['log_overallpercentofabnormalvolume'].min()
log_max = main_df['log_overallpercentofabnormalvolume'].max()

log_min_max = {'log_min': float(log_min), 'log_max': float(log_max)}

# Load log_min and log_max values from JSON file for inference
with open('log_min_max_pli_values.json', 'r') as infile:
    log_min_max_loaded = json.load(infile)
    log_min = log_min_max_loaded['log_min']
    log_max = log_min_max_loaded['log_max']


main_df['normalized_log_overallpercentofabnormalvolume'] = (main_df['log_overallpercentofabnormalvolume'] - log_min) / (log_max - log_min)

train_df = main_df.copy()
val_df = main_df.copy()


train_df['filepath'] = train_df['patient_id'].apply(get_training_image_path)

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                             xrv.datasets.XRayResizer(224)])

available_images_train_df = train_df[train_df['filepath'].apply(os.path.exists)]
print('Length of Train df: ', len(available_images_train_df))

train_dataset = MyDataset(available_images_train_df, transform=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


def get_validation_image_path(number):
    filename = f"{number}.png"
    path = DATA_PATH + 'valid/'
    filename = path+filename
    return filename

val_df['filepath'] = val_df['patient_id'].apply(get_validation_image_path)
available_images_val_df = val_df[val_df['filepath'].apply(os.path.exists)]
print('Length of validation df: ', len(available_images_val_df))

valid_dataset = MyDataset(available_images_val_df, transform=transforms)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)  # Adjust batch_size as appropriate


# Load pre-trained model and adjust for regression
model = xrv.models.DenseNet(weights="densenet121-res224-all")

# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = False

model.op_threshs = None  # Not needed for regression
model.classifier = torch.nn.Linear(1024, 1)  # Same as before, but now for regression

# Unfreeze the new classifier layer so it can be trained
for param in model.classifier.parameters():
    param.requires_grad = True

# Define optimizer - now only optimizing the classifier's parameters
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)  # Adjust learning rate as needed
criterion = torch.nn.MSELoss()  # Using Mean Squared Error Loss for regression
num_epochs = 100


best_val_mae = float('inf')  # Initialize best MAE as infinity
patience = 15  # Number of epochs to wait for improvement before stopping
patience_counter = 0  # Counter for epochs without improvement

# Track Model Progress on WandB platform
wandb.init(project="TBP_XRV_Regression", config={
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
},
    name="PLI-ensemble-regression-performance")


# Example training loop adjustment for regression
for epoch in range(num_epochs):
    model.train()
    train_loss = []
    for batch in train_dataloader:
        images = batch['img'].to(device)
        labels = batch['label'].to(device).float()  # Ensure labels are floats for regression
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()  # Squeeze to remove unnecessary dimensions
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
    avg_train_loss = sum(train_loss) / len(train_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation loop adjustment for regression
    model.eval()
    val_preds = []  # This will store the predictions after inverse transformations
    val_image_ids = []

    with torch.no_grad():
        for batch in valid_dataloader:
            images = batch['img'].to(device)
            outputs = model(images).squeeze()  # Get model predictions
            
            # Apply inverse transformations directly
            # First reverse the min-max scaling to get to log scale
            outputs_log_scale = inverse_min_max_scaling(outputs.cpu().numpy(), log_min, log_max)
            # Then apply the exponential function to reverse the logarithmic transformation
            outputs_original_scale = inverse_log_transformation(outputs_log_scale)

            val_preds.extend(outputs_original_scale)  # Now val_preds are in the original scale
            val_image_ids.extend(batch['image_id'])

    # Compute metrics
    original_labels = [main_df[main_df['patient_id'] == id]['overallpercentofabnormalvolume'].values[0] for id in val_image_ids]
    # Compute the average validation loss for this epoch
    val_mse = mean_squared_error(original_labels, val_preds)
    val_mae = mean_absolute_error(original_labels, val_preds)
    val_r2 = r2_score(original_labels, val_preds)
    wandb.log({"val_mae": val_mae, "val_mse": val_mse, "val_r2": val_r2}) 

    # Inside your training loop, after calculating validation MAE
    if val_mae < best_val_mae:
        print(f"Validation MAE improved from {best_val_mae:.4f} to {val_mae:.4f}. Saving model...")
        best_val_mae = val_mae
        patience_counter = 0  # Reset patience counter
        # Save model checkpoint
        torch.save(model.state_dict(), 'ensemble-pli-regression.pth')
    else:
        patience_counter += 1
        print(f"Validation MAE did not improve. Patience: {patience_counter}/{patience}")

    # Check if patience has been exceeded
    if patience_counter >= patience:
        print("Early stopping triggered. Stopping training...")
        break  # Exit the training loop


    # Print the validation metrics
    print(f"Epoch {epoch+1}/{num_epochs}, Validation MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, R-squared: {val_r2:.4f}")