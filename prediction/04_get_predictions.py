import os, csv, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import skimage
import torchvision
import torchxrayvision as xrv

parser = argparse.ArgumentParser()

parser.add_argument('-i', dest='CXR_PATH', type=str, required=True, help='Path to the chest x-ray (CXR) images')
parser.add_argument('--binary', dest='binary', action='store_true', help='If True, perform binary classification. Else, predict quantitative values with a binary model')
parser.add_argument('--timika', dest='get_timika_scores', action='store_true', help='Get Timika score predictions')
parser.add_argument('--pli', dest='get_pli', action='store_true', help='Get % lung involvement (PLI) predictions')

cmd_line_args = parser.parse_args()

CXR_PATH = cmd_line_args.CXR_PATH
binary = cmd_line_args.binary
get_timika_scores = cmd_line_args.get_timika_scores
get_pli = cmd_line_args.get_pli

MODEL_DIR = "../models"
DATA_DIR = "../data"

if not get_timika_scores and not get_pli:
    print(f"Please specify one of the flags --timika or --pli")
    exit()

if binary:
    model_suffix = 'binary-classifier'
else:
    model_suffix = 'regression'
    
if get_timika_scores:
    model_prefix = "timika"
elif get_pli:
    model_prefix = "pli"
    
MODEL_FILE = os.path.join(MODEL_DIR, f"{model_prefix}-ensemble-{model_suffix}.pth")
JSON_FILE = os.path.join(DATA_DIR, f"log_min_max_{model_prefix}_values.json")

assert os.path.isfile(MODEL_FILE)
assert os.path.isfile(JSON_FILE)

# force getting the basename because the file will be stored in the same directory. 
OUTPUT_CSV = os.path.join(CXR_PATH, f"{model_prefix}_{model_suffix}_predictions.csv")

print(f"Getting {model_prefix} predictions using model {MODEL_FILE} on images in {CXR_PATH} and saving to {OUTPUT_CSV}")

# Define paths
ALL_PNGS_PATH = f'{CXR_PATH}/png'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)  # Use xrv normalization
        img = img[None, ...]  # Add channel dimension

        if self.transform:
            img = self.transform(img)

        img = torch.from_numpy(img).float()
        patientID_view = os.path.splitext(img_name)[0]  # Extract patientID_view (i.e. T0001_1A)
        return {'img': img, 'patientID_view': patientID_view}

# Define transformations using torchxrayvision
transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),  # Center crops the image
    xrv.datasets.XRayResizer(224)   # Resizes the image to 224x224
])

# Prepare Dataset and DataLoader
dataset = MyDataset(image_dir=ALL_PNGS_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load Model
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.op_threshs = None
model.classifier = torch.nn.Linear(1024, 1)  # Update output layer for regression
model.load_state_dict(torch.load(MODEL_FILE))
model.to(device)
model.eval()

if binary:
    
    # Generate predictions
    output_data = []
    with torch.no_grad():
        
        for batch in dataloader:
            images = batch['img'].to(device)
            image_ids = batch['patientID_view']

            outputs = model(images).squeeze().cpu().numpy()

            # Apply sigmoid to convert logits to probabilities
            probabilities = 1 / (1 + np.exp(-outputs))
            binary_preds = (probabilities > 0.5).astype(int)

            for patientID_view, pred in zip(image_ids, binary_preds):
                output_data.append([patientID_view, pred])

    # Write predictions to CSV
    df_output = pd.DataFrame(output_data)
    df_output.columns = ['patientID_view', 'predicted_binary_label']
    df_output.to_csv(OUTPUT_CSV, index=False)

else:
    
    # Load log transform and min-max scaling values
    with open(JSON_FILE, 'r') as infile:
        log_min_max_loaded = json.load(infile)
        log_min = log_min_max_loaded['log_min']
        log_max = log_min_max_loaded['log_max']

    def inverse_min_max_scaling(norm_value, log_min, log_max):
        return norm_value * (log_max - log_min) + log_min

    def inverse_log_transformation(log_value):
        return np.exp(log_value) - 1

    # Generate predictions and store them
    output_data = []

    with torch.no_grad():

        for batch in dataloader:
            images = batch['img'].to(device)
            patient_ids = batch['patientID_view']

            # Predict
            outputs = model(images).squeeze().cpu().numpy()
            outputs_log_scale = inverse_min_max_scaling(outputs, log_min, log_max)
            outputs_original_scale = inverse_log_transformation(outputs_log_scale)

            for patientID_view, pred in zip(patient_ids, outputs_original_scale):
                output_data.append([patientID_view, pred])

    # Write predictions to CSV
    df_output = pd.DataFrame(output_data)
    df_output.columns = ['patientID_view', 'predicted_label']
    df_output.to_csv(OUTPUT_CSV, index=False)