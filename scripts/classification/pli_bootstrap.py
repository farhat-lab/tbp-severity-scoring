from glob import glob
from datetime import date

import os
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import torch.nn as nn
import skimage.io
import wandb
import torchxrayvision as xrv


# Specify Project details

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, '../data')
IMAGE_METADATA_PATH = os.path.join(DATA_PATH, "tbp_cxr_metadata.csv")  # sample CSV file containing labels
MODEL_PATH = os.path.join(ABS_PATH, "../model/pli-ensemble-binary-classifier.pth")


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
        label = self.dataframe.iloc[idx]['PLI_binary_var']  # Adjust 'label' to the name of your actual label column

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

# Assuming the device setup and model architecture are the same as in the training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.op_threshs = None
model.classifier = torch.nn.Linear(1024, 1)  # Adjust for your specific model
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()


# Bootstrapping for AUC
n_bootstrap_samples = 100
auc_scores = []

for _ in range(n_bootstrap_samples):
    sampled_preds, sampled_labels = [], []
    for batch in test_dataloader:
        images = batch['img'].to(device)
        labels = batch['label'].numpy()  # assuming you want to process labels on CPU

        with torch.no_grad():
            outputs = model(images).squeeze().cpu().numpy()
        sampled_preds.extend(outputs)
        sampled_labels.extend(labels)

    resampled_preds, resampled_labels = resample(sampled_preds, sampled_labels)
    auc_score = roc_auc_score(resampled_labels, resampled_preds)
    auc_scores.append(auc_score)

# Calculate 95% confidence interval
lower = np.percentile(auc_scores, 2.5)
upper = np.percentile(auc_scores, 97.5)
print(f"AUC: {np.mean(auc_scores):.4f} (95% CI: {lower:.4f}-{upper:.4f})")

