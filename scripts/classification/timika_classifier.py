from glob import glob
from datetime import date, datetime
from sklearn import metrics
from torch.utils.data import Dataset, Subset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch  
import torchvision
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
        label = self.dataframe.iloc[idx]['timika_binary_var']  # Adjust 'label' to the name of your actual label column

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

    
def get_training_image_path(number):
    filename = f"{number}.png"
    path = DATA_PATH + 'train/'
    filename = path+filename
    return filename

main_df = pd.read_csv(IMAGE_METADATA_PATH)
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

# Load pre-trained model and erase classifier
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.op_threshs = None # prevent pre-trained model calibration

for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)  # num_classes is your task-specific class count

optimizer = torch.optim.Adam(model.classifier.parameters()) # default LR = 0.001
criterion = torch.nn.BCEWithLogitsLoss()

num_epochs = 100  # Example number of epochs

# Track Model Progress on WandB platform
wandb.init(project="TBP_XRV_Regression", config={
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
},
    name="timika_ensemble_binary_classifier-performance")

# Early Stopping Setup
best_val_auc = float("-inf")  # Initialize the best AUC score as the lowest possible score
patience = 15  # Number of epochs to wait for improvement before stopping
epochs_since_improvement = 0  # Counter for the epochs since last improvement

for epoch in range(num_epochs):
    # Initialize accumulators

    correct_predictions = 0
    total_predictions = 0

    model.train()  # Set model to training mode
    train_loss = []
    for batch in train_dataloader:
        images = batch['img'].to(device)
        labels = batch['label'].to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)

         # Assuming binary classification with a sigmoid output
        predicted_labels = (torch.sigmoid(outputs) > 0.5).float()  # Apply threshold
        correct_predictions += (predicted_labels == labels.unsqueeze(1)).sum().item()
        total_predictions += labels.size(0)

        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())

    # Calculate training accuracy
    train_accuracy = correct_predictions / total_predictions

    # Calculate average training loss for the epoch
    avg_train_loss = sum(train_loss) / len(train_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {train_accuracy:.4f}, Training Loss: {avg_train_loss:.4f}")

    # Log training metrics
    wandb.log({"PLI Training Accuracy": train_accuracy, "Epoch": epoch+1})

    
    # Validation phase
    model.eval()  # Set model to evaluation mode
    # Initialize or reset metrics to track during validation
    val_auc_score = 0 
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for batch in valid_dataloader:
            images = batch['img'].to(device)
            labels = batch['label'].to(device).float()
            val_labels.extend(labels.cpu().numpy())
            
            outputs = model(images)
            predictions = torch.sigmoid(outputs).detach().cpu().numpy()[:, 0]
            val_preds.extend(predictions)


    # After validation predictions are collected
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    val_pred_labels = (val_preds > 0.5).astype(int)  # Assuming a threshold of 0.5 for binary classification

    # Compute Precision-Recall curve
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(val_labels, val_preds)

    # Calculate F1 Score
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Compute AUC Score
    val_auc_score = sklearn.metrics.roc_auc_score(val_labels, val_preds)

    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(val_labels, val_preds)
    # roc_auc = sklearn.metrics.auc(fpr, tpr)

    # Compute Accuracy
    val_accuracy = sklearn.metrics.accuracy_score(val_labels, val_pred_labels)

    # Compute confusion matrix to extract true positives, false positives, true negatives, and false negatives
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(val_labels, val_pred_labels).ravel()

    # Calculate Sensitivity (Recall) and Specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)


    # Calculate validation metrics properly outside the loop
    
    # Early Stopping and Model Saving Logic
    if val_auc_score > best_val_auc:
        print(f"Validation AUC improved from {best_val_auc:.4f} to {val_auc_score:.4f}. Saving model...")
        best_val_auc = val_auc_score  # Update the best known AUC score
        epochs_since_improvement = 0  # Reset the counter
        # Save the model
        torch.save(model.state_dict(), 'timika_ensemble_binary_classifier.pth')
        # Optionally log the new best AUC to wandb
        wandb.log({"Best Validation AUC": best_val_auc})
    else:
        epochs_since_improvement += 1
        print(f"Validation AUC did not improve. Current patience: {epochs_since_improvement}/{patience}.")
    
    if epochs_since_improvement >= patience:
        print("Early stopping triggered. Stopping training.")
        break  # Exit from the training loop


    # Calculate metrics (e.g., AUC) after collecting all predictions and labels for the validation set
    print(f"Epoch: {epoch+1}/{num_epochs}, Validation AUC Score: {val_auc_score:.4f}, Validation Accuracy: {val_accuracy:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")

    # Log validation metrics
    wandb.log({"Timika Validation AUC": val_auc_score,
               "Timika Validation Accuracy": val_accuracy,
               "Epoch": epoch+1})
    