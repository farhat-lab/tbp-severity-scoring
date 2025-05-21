# Get CXR Predictions on New Data

The goal of the scripts in this directory is to obtain CXR (percent lung involvement, PLI, and Timika score) predictions for new chest x-ray DICOM images.

To run this pipeline, please move all DICOM images into a directory called `dicom` within another directory. The top-level directory will be where output files (like the predictions) will be saved. For example, you can have a directory called `new_CXR_predictions`. Then all the DICOM images must be in `new_CXR_predictions/dicom`.

There are 4 scripts to run, in the numbered order:

## `01_dicom_metadata_extraction.py`

This script extracts metadata from DICOM messages. The metadata includes information about the patient, referring physician, date, time, and location that the x-ray was taken.

Required arguments:

<ul>
    <li><code>CXR_PATH</code>: Directory with DICOM images. The DICOM files should be in a subdirectory called 'dicom'</li>
</ul>

## `02_outlier_detector_using_dicoms.py`

This script extracts metadata from DICOM messages. The metadata includes information about the patient, referring physician, date, time, and location that the x-ray was taken.

Required arguments:

<ul>
    <li><code>CXR_PATH</code>: Directory with DICOM images. The DICOM files should be in a subdirectory called 'dicom'</li>
</ul>

## `03_PNG_conversion_and_comparison.py`

This script extracts metadata from DICOM messages. The metadata includes information about the patient, referring physician, date, time, and location that the x-ray was taken.

Required arguments:

<ul>
    <li><code>CXR_PATH</code>: Directory with DICOM images. The DICOM files should be in a subdirectory called 'dicom'</li>
</ul>

## `04_get_predictions .py`

This script extracts metadata from DICOM messages. The metadata includes information about the patient, referring physician, date, time, and location that the x-ray was taken.

Required arguments:

<ul>
    <li><code>CXR_PATH</code>: Directory with DICOM images. The DICOM files should be in a subdirectory called <code>dicom</code></li>
    <li><code>binary</code>: If True, perform binary classification (uncommon because it has been pre-binarized at a threshold). Else, predict quantitative values with a binary model </li>
</ul>

You must then specify either `--timika` or `--pli` to get predictions using one of the models.

# Create Virtual Environment

```bash
cd prediction
env_name="CXR_predictions" # can change this if desired
conda env create -f venv.yaml --name $env_name

# then install pytorch using pip. It often gets killed because it requires too much RAM if you include these pip packages in the YAML file above.
source activate $env_name
pip install torch==2.4.1+cpu torchaudio==2.4.1+cpu torchvision==0.19.1+cpu torchxrayvision==1.3.3 --extra-index-url https://download.pytorch.org/whl/cpu
```

# Get Predictions

To run the full pipeline, run the following from the command line. You will have to run `04_get_predictions.py` twice, once for Timika score predictions and once for percent lung involvement predictions.

There is a file of 10 DICOM images to test getting predictions for.

```bash
cd prediction
env_name="CXR_predictions"
CXR_PATH="test_images"
source activate $env_name

python3 -u 01_dicom_metadata_extraction.py -i $CXR_PATH
python3 -u 02_outlier_detector_using_dicoms.py -i $CXR_PATH
python3 -u 03_PNG_conversion_and_comparison.py -i $CXR_PATH

python3 -u 04_get_predictions.py -i $CXR_PATH --timika
python3 -u 04_get_predictions.py -i $CXR_PATH --pli
```