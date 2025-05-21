import argparse
import numpy as np
import os
import pydicom
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-i', dest='CXR_PATH', type=str, required=True, help='Path to the chest x-ray (CXR) images')

cmd_line_args = parser.parse_args()

CXR_PATH = cmd_line_args.CXR_PATH
dicom_PATH = os.path.join(CXR_PATH, "dicom")


def extract_metadata(dicom_file):
    """
    Function to extract metadata from a DICOM file, avoiding sequences.
    """
    metadata = {}
    for elem in dicom_file.iterall():
        if elem.VR != 'SQ':  # Avoid sequences
            metadata[elem.keyword] = elem.value
    return metadata


def parse_filename(filename):
    """
    Function to parse the filename to get patient_id and view type with error handling.
    """
    
    base_name = os.path.basename(filename).split('.')[0]  # Remove file extension
    
    if base_name.count('_') < 1 or base_name.count('_') > 2:
        raise ValueError(f"Filename {base_name} does not conform to expected pattern <patient_id>_<view>.dcm")
    
    patient_id = base_name.split('_')[0]
    view = base_name.split('_')[1]
    
    # means there are two images per view, with names like pid_view_1 and pid_view_2
    if base_name.count('_') == 2:
        file_num = base_name.split('_')[2]
    else:
        file_num = 1
    
    return patient_id, view, file_num


# Traverse the directory to process DICOM files
num_files = len(os.listdir(dicom_PATH))
print(f"{num_files} total files")

# Lists to hold data for the DataFrame
patient_ids = []
views = []
file_nums = []
metadata_list = []

for root, _, files in os.walk(dicom_PATH):
    
    for i, file in enumerate(files):
        
        if file.lower().endswith('.dcm'):
            file_path = os.path.join(root, file)
            dicom = pydicom.dcmread(file_path)
            metadata = extract_metadata(dicom)
            
            patient_id, view, file_num = parse_filename(file)
            
            patient_ids.append(patient_id)
            views.append(view)
            file_nums.append(int(file_num))
            
            metadata_list.append(metadata)
            
        if i % 100 == 0:
            print(f"Finished file {i+1}/{num_files}")

# Create DataFrame with metadata
df = pd.DataFrame(metadata_list)
df['patient_id'] = patient_ids
df['view'] = views
df['image_replicate'] = file_nums

assert len(df) == num_files

if not os.path.isfile(os.path.join(CXR_PATH, 'cleaned_dicom_metadata.csv.gz')):
    df[['patient_id', 'view', 'image_replicate', 'Modality', 'ViewPosition', 'PatientOrientation', 'PhotometricInterpretation', 'Rows', 'Columns', 'PixelData']].to_csv(os.path.join(CXR_PATH, 'cleaned_dicom_metadata.csv.gz'), index=False, compression='gzip')

# IMAGE DISPLAY CODE:
# # Filter to include only specific views
# unique_view_types = ['1A', '1B', '1C', '1D', '2A', '2B', '2C']
# unique_df = cleaned_df[cleaned_df['view'].isin(unique_view_types)]

# # Display 2 images from each unique view (1A, 1B, 1C, 1D, 2A, 2B, 2C) per patient
# images_displayed = {}  # Dictionary to keep track of images displayed per view

# for i, patient_id in enumerate(unique_df['patient_id'].unique()):
    
#     patient_df = unique_df[unique_df['patient_id'] == patient_id]
    
#     for view in unique_view_types:
#         view_df = patient_df[patient_df['view'] == view]
        
#         if not view_df.empty:
#             # Initialize counter for each view type if not already initialized
#             if view not in images_displayed:
#                 images_displayed[view] = 0
            
#             # Limit display to 2 images per view type
#             for index, row in view_df.iterrows():
#                 if images_displayed[view] < 2:
                    
#                     if row['image_replicate'] != 1:
#                         dicom_file_path = os.path.join(dicom_PATH, f"{row['patient_id']}_{row['view']}_{row['image_replicate']}.dcm")
#                     else:
#                         dicom_file_path = os.path.join(dicom_PATH, f"{row['patient_id']}_{row['view']}.dcm")
                        
#                     if os.path.exists(dicom_file_path):
#                         dicom = pydicom.dcmread(dicom_file_path)
#                         plt.imshow(dicom.pixel_array, cmap='gray')
#                         plt.title(f"Patient ID: {row['patient_id']} View: {row['view']}")
#                         plt.axis('off')
#                         plt.show()
#                         images_displayed[view] += 1

#                 else:
#                     break
             
#     # only show the first 10 patients to avoid crashing the notebook
#     if i == 10:
#         break
