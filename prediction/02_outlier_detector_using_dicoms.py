import csv, argparse, sys
from pathlib import Path

# tbpcxr folder, which is a subfolder of tbp-cxr-outlier has necessary functions
sys.path.append("tbp-cxr-outlier")
from tbpcxr.model import Model
from tbpcxr.utilities import read_dcm

parser = argparse.ArgumentParser()

parser.add_argument('-i', dest='CXR_PATH', type=str, required=True, help='Path to the chest x-ray (CXR) image folder')

cmd_line_args = parser.parse_args()

CXR_PATH = cmd_line_args.CXR_PATH

# Path to the directory containing DICOM files
dicom_directory = Path(f"{CXR_PATH}/dicom")

# Get a list of all DICOM files in the directory
image_file_list = [p for p in dicom_directory.iterdir() if p.suffix == '.dcm']
print(f"{len(image_file_list)} images")

# Load the pre-trained outlier detection model
outlier_model = Model.load_outlier_pcamodel()

# Prepare observations
observations = []
valid_files = []

for fn in image_file_list:
    try:
        # Read each DICOM file
        obs = read_dcm(fn)
        observations.append(obs)
        valid_files.append(fn)  # Only add to valid list if read_dcm is successful
    except Exception as e:
        print(f"Error reading {fn}: {e}")

# Process the observations if there are valid files
if observations:
    try:
        arr = outlier_model.to_observations(observations)
        results = outlier_model.outlier_predictor(arr)

        # Prepare the output data for writing into CSV
        output_data = [(str(fn), 'Outlier' if o == -1 else 'Normal') for fn, o in zip(valid_files, results)]

        # Define the path for the output CSV file
        csv_file_path = f'{CXR_PATH}/outlier_detection_results_all_dicoms.csv'

        # Write results to a CSV file
        with open(csv_file_path, mode='w+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Filename', 'Status'])  # Write header
            writer.writerows(output_data)

        print(f"Results written to {csv_file_path}")
    except Exception as e:
        print(f"Error processing valid observations: {e}")
else:
    print("No valid observations found.")