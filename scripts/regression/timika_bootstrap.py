from glob import glob
from datetime import date
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import json
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
import torchvision
import torchxrayvision as xrv
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

# Project details and data paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, '../data')
MODEL_PATH = os.path.join(ABS_PATH, "../model/ensemble-timika-regression.pth")
IMAGE_METADATA_PATH = os.path.join(DATA_PATH, "tbp_cxr_metadata.csv")  # sample CSV file containing labels
JSON_FILE = os.path.join(DATA_PATH, "log_min_max_timika_values.json")  # JSON file containing values to inverse the log transform and min-max scaling



class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['normalized_log_timika']
        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)
        img = img[None, ...]

        if self.transform:
            img = self.transform(img)

        img = torch.from_numpy(img).float()
        label = torch.tensor(label, dtype=torch.float)
        image_id = self.dataframe.iloc[idx]['patient_id']

        return {'img': img, 'label': label, 'image_id': image_id}

main_df = pd.read_csv(IMAGE_METADATA_PATH)
main_df['log_timika'] = np.log1p(main_df['timika'])

with open(JSON_FILE, 'r') as infile:
    log_min_max_loaded = json.load(infile)
    log_min = log_min_max_loaded['log_min']
    log_max = log_min_max_loaded['log_max']

main_df['normalized_log_timika'] = (main_df['log_timika'] - log_min) / (log_max - log_min)
test_df = main_df.copy()

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

def get_test_image_path(number):
    filename = f"{number}.png"
    path = DATA_PATH + 'test/'   # Add files to the test set to test model performance
    return os.path.join(path, filename)

test_df['filepath'] = test_df['patient_id'].apply(get_test_image_path)
available_images_test_df = test_df[test_df['filepath'].apply(os.path.exists)]
print('Length of test df: ', len(available_images_test_df))

test_dataset = MyDataset(available_images_test_df, transform=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.op_threshs = None
model.classifier = torch.nn.Linear(1024, 1)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

def inverse_min_max_scaling(norm_value, log_min, log_max):
    return norm_value * (log_max - log_min) + log_min

def inverse_log_transformation(log_value):
    return np.exp(log_value) - 1

# Collecting predictions and labels
sampled_preds, sampled_labels, sampled_image_ids = [], [], []
for batch in test_dataloader:
    images = batch['img'].to(device)
    labels = batch['label'].cpu().numpy()
    image_ids = batch['image_id']
    with torch.no_grad():
        outputs = model(images).squeeze().cpu().numpy()
        outputs_log_scale = inverse_min_max_scaling(outputs, log_min, log_max)
        outputs_original_scale = inverse_log_transformation(outputs_log_scale)
        
    sampled_preds.extend(outputs_original_scale)
    sampled_image_ids.extend(image_ids)

# Fetch the original labels using image_id
sampled_labels = [main_df[main_df['patient_id'] == id]['timika'].values[0] for id in sampled_image_ids]

# Bootstrapping using sklearn.utils.resample
n_bootstrap_samples = 500
metrics = {'mae': [], 'mse': [], 'r2': []}

for _ in range(n_bootstrap_samples):
    # Resampling with replacement
    resampled_preds, resampled_labels = resample(sampled_preds, sampled_labels)
    
    # Calculate metrics
    metrics['mae'].append(mean_absolute_error(resampled_labels, resampled_preds))
    metrics['mse'].append(mean_squared_error(resampled_labels, resampled_preds))
    metrics['r2'].append(r2_score(resampled_labels, resampled_preds))

# Calculate 95% confidence intervals
for metric_name, scores in metrics.items():
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    print(f"{metric_name.upper()}: {np.mean(scores):.4f} (95% CI: {lower:.4f}-{upper:.4f})")
