# Convert RGB DICOMs to single channel Grayscale pngs
# Compare image quality of the converted pngs with the DICOMs

# what happens if you predict on the outliers?
# do you make more mistakes relative to human PLI, unilateral/bilateral, or cavitation?
# Run all the models normal, outliers and compare with human reads

import os, argparse, glob
import pandas as pd
import pydicom
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('-i', dest='CXR_PATH', type=str, required=True, help='Path to the chest x-ray (CXR) images')

cmd_line_args = parser.parse_args()

CXR_PATH = cmd_line_args.CXR_PATH


def convert_rgb_dicom_to_png(dicom_dir, png_dir):
    
    os.makedirs(png_dir, exist_ok=True)  # Ensure the output folder exists

    mapping_dict = {}  # Dictionary to record mappings
    error_paths = []  # List to capture paths with issues

    # Iterate over each row in the DataFrame
    for dicom_filename in glob.glob(f"{dicom_dir}/*.dcm"):
        
        # Generate PNG filename from patient_id and view
        png_filename = os.path.join(png_dir, os.path.basename(dicom_filename).replace('.dcm', '.png'))
        print(f"Converting {dicom_filename} to {png_filename}")
        
        try:
            # Read the DICOM file
            dicom_data = pydicom.dcmread(dicom_filename)

            # Extract RGB pixel array
            image_3d = dicom_data.pixel_array

            # Normalize pixel values if necessary
            max_val = np.max(image_3d)
            if max_val > 0:  # Normalize only if max > 0 to avoid division by zero
                image_3d = (image_3d / max_val) * 255.0

            # Convert to uint8
            image_3d = image_3d.astype(np.uint8)

            # Convert RGB to grayscale
            # Using the formula Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
            image_gray = np.dot(image_3d[...,:3], [0.2989, 0.5870, 0.1140])
            image_gray = image_gray.astype(np.uint8)

            # Save as PNG
            Image.fromarray(image_gray).save(png_filename)

        except Exception as e:
            print(f"Error converting {dicom_filename}: {e}")


# Run conversion
convert_rgb_dicom_to_png(f"{CXR_PATH}/dicom", f"{CXR_PATH}/png")