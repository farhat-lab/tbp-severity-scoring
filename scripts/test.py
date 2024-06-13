from glob import glob
from datetime import date, datetime
from sklearn import metrics
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample

import json
import matplotlib.pyplot as plt
import numpy as np
import pdb
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

# Specify Project details

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, '../data')
MODEL_PATH = os.path.join(ABS_PATH, "../model/ensemble-pli-regression.pth")
IMAGE_METADATA_PATH = os.path.join(DATA_PATH, "tbp_cxr_metadata.csv")  # sample CSV file containing labels
JSON_FILE = os.path.join(DATA_PATH, "log_min_max_pli_values.json")  # Load log_min and log_max values from JSON file for reversing the log transform and rescaling

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


main_df = pd.read_csv(IMAGE_METADATA_PATH)
main_df['log_overallpercentofabnormalvolume'] = np.log1p(main_df['overallpercentofabnormalvolume'])


# Load log_min and log_max values from JSON file for inference
with open(JSON_FILE, 'r') as infile:
    log_min_max_loaded = json.load(infile)
    log_min = log_min_max_loaded['log_min']
    log_max = log_min_max_loaded['log_max']


main_df['normalized_log_overallpercentofabnormalvolume'] = (main_df['log_overallpercentofabnormalvolume'] - log_min) / (log_max - log_min)
test_df = main_df.copy()

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                             xrv.datasets.XRayResizer(224)])


def get_test_image_path(number):
    filename = f"{number}.png"
    path = DATA_PATH + 'test/'   # Add files to the test set to test model performance
    return os.path.join(path, filename)


test_df['filepath'] = test_df['patient_id'].apply(get_test_image_path)
available_images_test_df = test_df[test_df['filepath'].apply(os.path.exists)]
print('Length of test df: ', len(available_images_test_df))

test_dataset = MyDataset(available_images_test_df, transform=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Adjust batch_size as appropriate

# The device setup and model architecture are the same as in the training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.op_threshs = None
model.classifier = torch.nn.Linear(1024, 1)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()


# Function to reverse transformations for evaluation
def inverse_min_max_scaling(norm_value, log_min, log_max):
    return norm_value * (log_max - log_min) + log_min

def inverse_log_transformation(log_value):
    return np.exp(log_value) - 1



# Collect predictions and labels from the test set
val_preds, val_image_ids = [], []
for batch in test_dataloader:
    images = batch['img'].to(device)
    labels = batch['label'].cpu().numpy()  # Extract and store true labels
    with torch.no_grad():
        outputs = model(images).squeeze().cpu().numpy()
        outputs_log_scale = inverse_min_max_scaling(outputs, log_min, log_max)
        outputs_original_scale = inverse_log_transformation(outputs_log_scale)

        val_preds.extend(outputs_original_scale)
        val_image_ids.extend(batch['image_id'])

# Compute original labels using the patient IDs
original_labels = [main_df[main_df['patient_id'] == id]['overallpercentofabnormalvolume'].values[0] for id in val_image_ids]

# Calculate metrics
mae = mean_absolute_error(original_labels, val_preds)
mse = mean_squared_error(original_labels, val_preds)
r2 = r2_score(original_labels, val_preds)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
