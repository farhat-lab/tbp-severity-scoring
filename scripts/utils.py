from datetime import date, datetime
from matplotlib import pyplot as plt
from PIL import Image

import collections
import csv
import os
import pickle
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def convert_dicom_to_png(csv_path, dicom_col, patient_id_col, output_folder, mapping_csv_path):
    # Create the output folder to store the PNG images
    os.makedirs(output_folder, exist_ok=True)

    # Create a dictionary to store the mapping between DICOM path and PNG filename
    mapping_dict = {}
    # Create a list to store the paths that generate FileNotFoundError
    error_paths = []
    # Create a list to store the paths with missing pixel data
    missing_pixel_data_paths = []
    # Create a list to store the paths that generate ValueError
    value_error_paths = []

    df = pd.read_csv(csv_path)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        dicom_path = row[dicom_col]
        patient_id = row[patient_id_col]

        # Generate the PNG filename based on the patient ID
        png_filename = f'{patient_id}.png'

        # Add the mapping to the dictionary
        mapping_dict[dicom_path] = png_filename

        try:
            # Read the DICOM file
            dicom_data = pydicom.dcmread(CXR_IMAGE_PATH + dicom_path)

            # Check if the required elements are present in the DICOM dataset
            if 'PixelData' not in dicom_data:
                # Store the path with missing pixel data
                missing_pixel_data_paths.append(dicom_path)
                continue

             # Convert to float to avoid overflow or underflow losses
            image_2d = dicom_data.pixel_array.astype(float)
            # Rescale the grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d, 0) / np.max(image_2d)) * 255.0
            # Invert pixel values for MONOCHROME1 images
            invert_image_2d_scaled = 255 - image_2d_scaled  # remove this line for MONOCHROME2 images
            # Convert to uint8
            final_image_2d_scaled = np.uint8(invert_image_2d_scaled)


            # Save as PNG
            png_path = os.path.join(output_folder, png_filename)
            Image.fromarray(final_image_2d_scaled).save(png_path)


        except FileNotFoundError:
            # Store the path that generates FileNotFoundError
            error_paths.append(dicom_path)

        except ValueError:
            # Store the path that generates ValueError
            value_error_paths.append(dicom_path)
            continue



def inverse_min_max_scaling(norm_value, log_min, log_max):
    return norm_value * (log_max - log_min) + log_min

def inverse_log_transformation(log_value):
    return np.exp(log_value) - 1
