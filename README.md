# tbp-severity-scoring

## Part 2: CNN
This part contains the code and data used for the analysis of chest X-ray images (CXRs) for tuberculosis (TB) patients using the TB Portals Program (TBP) database.

## Repository Structure
- `data/`: Contains the processed data files used for the analysis.
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


