# Medical Insurance Cost Prediction

This project builds a machine learning pipeline to predict medical insurance charges from demographic and health-related features. It includes data loading utilities, exploratory analysis, model training, and a simple deployment-ready app structure.

## Project Overview

The goal is to estimate insurance costs based on attributes such as:
- age
- sex
- bmi
- children
- smoker status
- region

The project uses a Python-based workflow with pandas, scikit-learn, matplotlib, seaborn, plotly, and streamlit.

## Project Structure

- `data/` - dataset files
- `notebooks/` - Jupyter notebooks for analysis and experimentation
- `src/` - reusable Python modules and utilities
- `models/` - trained model artifacts
- `outputs/` - generated reports, figures, and metrics
- `app/` - application entry points
- `tests/` - test cases

## Requirements

The project dependencies are listed in `requirements.txt`.

## Setup

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run the utilities check

```powershell
$env:PYTHONPATH="."
python -c "from src.utils import load_raw_data, inspect_dataframe; df = load_raw_data(); inspect_dataframe(df)"
```

## Dataset

Place the dataset in `data/insurance.csv` before running the pipeline.

## Usage

You can use the project in the following ways:
- explore the data in notebooks
- train and evaluate models from the training scripts
- run the app once the interface is implemented

## Notes

- The repository ignores local environment and output artifacts such as `.venv/`, `__pycache__/`, and generated files in `outputs/`.
- Keep the dataset local and do not commit large data files unless required.

## License

This project is for educational and personal use.
