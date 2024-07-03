# tbp-severity-scoring

## Part 2: CNN
This repository contains the code and sample data used for the research work titled "Percent lung involved with tuberculosis on chest X-ray predicts unfavorable treatment outcome and is accurately predicted with artificial intelligence". More information on the raw data used for this research work is present in the TB-Portals website (https://tbportals.niaid.nih.gov).

## Repository Structure
- `data/`: Contains the example data files used for the analysis.
- `models/`: Ensemble model weights.
- `notebooks/`: Jupyter notebooks for cohort selection, quality checking and data exploration.
- `scripts/`: Python scripts for regression and classification tasks.
- `requirements.txt`: Python dependencies required to run the scripts.
- `LICENSE`: License information.

## Getting Started
### Prerequisites
- Python 3.7 or higher
- Git
- Virtual environment tools (optional)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/farhat-lab/tbp-severity-scoring.git
cd tbp-severity-scoring
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```


## Usage
To test the PLI regression model on your image samples:
```bash
python scripts/test.py
```

## License Notice 
The model weights are deposited for peer-review purposes only. No license is provided for reuse of this repository at this time. 
