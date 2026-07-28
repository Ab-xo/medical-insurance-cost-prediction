# Medical Insurance Cost Prediction

An end-to-end machine learning regression project that predicts annual medical insurance charges based on demographic and health attributes such as age, BMI, smoking status, number of dependants, sex, and region.

## Problem Statement

Insurance companies need to price policies accurately — underpricing causes financial loss, overpricing drives customers away. This project builds a regression pipeline that:

1. Cleans and preprocesses raw policyholder data
2. Engineers meaningful features (age groups, BMI categories, smoker × obesity interaction)
3. Trains and compares 10 regression algorithms
4. Deploys the best model via an interactive Streamlit application

**Target variable:** `charges` (USD)

---

## Dataset

| Attribute  | Description                                                                |
| ---------- | -------------------------------------------------------------------------- |
| `age`      | Age of the primary beneficiary                                             |
| `sex`      | Gender — `male` / `female`                                                 |
| `bmi`      | Body Mass Index (kg/m²)                                                    |
| `children` | Number of dependants covered by the policy                                 |
| `smoker`   | Smoking status — `yes` / `no`                                              |
| `region`   | US residential region — `northeast`, `northwest`, `southeast`, `southwest` |
| `charges`  | Annual medical insurance cost in USD — **target**                          |

- **Records:** 1,338 policyholders (1,337 after deduplication)
- **Features:** 6 input + 1 target
- **Missing values:** None
- **Source:** [Kaggle — Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset/data)

---

## Installation

### Prerequisites

- Python 3.11+ (tested on 3.13)
- pip

### Setup

```bash
git clone https://github.com/Ab-xo/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### 1. Inspect the Dataset

```powershell
# Windows PowerShell
$env:PYTHONPATH = "."
python -c "from src.utils import load_raw_data, generate_data_understanding_report; generate_data_understanding_report(load_raw_data())"
```

Outputs a full data understanding report to `outputs/reports/data_understanding.md`.

### 2. Train Models

```powershell
$env:PYTHONPATH = "."
python -m src.train
```

Runs the full pipeline: clean → engineer features → split → train 10 models → cross-validate → save best model → write reports.

### 3. Make Predictions (CLI)

```python
from src.predict import predict_charges

sample = {
    "age": 35,
    "sex": "male",
    "bmi": 28.5,
    "children": 2,
    "smoker": "no",
    "region": "northwest",
}

charges = predict_charges(sample)
print(f"Predicted charges: ${charges:,.2f}")
```

### 4. Launch Streamlit App

```bash
streamlit run app/app.py
```

### 5. Run Tests

```powershell
$env:PYTHONPATH = "."
pytest tests/ -v
```

### 6. Explore EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Project Structure

```
medical-insurance-cost-prediction/
├── data/
│   └── insurance.csv              # Raw dataset (1,338 rows)
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Paths, logger, data inspection, I/O
│   ├── preprocessing.py           # Cleaning, validation, stratified split
│   ├── feature_engineering.py     # 8 domain-driven derived features
│   ├── train.py                   # Full training pipeline (10 models)
│   ├── evaluate.py                # Metrics, CV, learning curves, model selection
│   ├── predict.py                 # Inference
│   └── visualization.py           # EDA & evaluation plots
├── models/
│   ├── best_model.pkl             # Saved best model pipeline (generated)
│   ├── feature_info.json          # Feature metadata for inference
│   └── model_metadata.json        # Best model name + config
├── outputs/
│   ├── figures/
│   │   ├── eda/                   # EDA visualisations (generated)
│   │   └── evaluation/            # Model eval plots (generated)
│   ├── metrics/                   # comparison.csv, cross_validation.json
│   └── reports/                   # Markdown reports, cleaned_data.csv
├── app/
│   └── app.py                     # Streamlit web application
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Results

### Model Comparison

| Rank | Model                       | RMSE       | MAE        | R²         | CV R² | Train Time |
| ---- | --------------------------- | ---------- | ---------- | ---------- | ----- | ---------- |
| ⭐ 1 | **Lasso Regression**        | **$4,502** | **$2,438** | **0.8852** | 0.851 | 0.08s      |
| 2    | Linear Regression           | $4,503     | $2,439     | 0.8852     | 0.851 | 0.03s      |
| 3    | Ridge Regression            | $4,552     | $2,545     | 0.8827     | 0.848 | 0.10s      |
| 4    | XGBoost Regressor           | $4,703     | $2,456     | 0.8747     | 0.842 | 0.86s      |
| 5    | Gradient Boosting Regressor | $4,743     | $2,451     | 0.8726     | 0.838 | 1.42s      |
| 6    | Random Forest Regressor     | $4,823     | $2,617     | 0.8683     | 0.825 | 1.05s      |
| 7    | Extra Trees Regressor       | $4,963     | $2,693     | 0.8605     | 0.813 | 0.60s      |
| 8    | Support Vector Regressor    | $5,100     | $2,000     | 0.8527     | 0.811 | 0.32s      |
| 9    | AdaBoost Regressor          | $5,723     | $4,866     | 0.8145     | 0.787 | 1.32s      |
| 10   | Decision Tree Regressor     | $6,396     | $3,207     | 0.7683     | 0.744 | 0.10s      |

**Best Model:** Lasso Regression — selected via composite ranking (RMSE 40%, MAE 35%, R² 15%, training time 10%). L1 regularisation naturally performs feature selection by shrinking weak coefficients to zero, which fits this dataset where smoker status dominates the signal and other features contribute smaller marginal effects.

### Key Findings

- **Smoker status** is the dominant predictor — smokers pay **3.8× more** on average ($32,050 vs $8,434)
- **BMI ≥ 30 combined with smoking** creates the highest-charge group — the `smoker_obese` interaction captures this compounding risk
- **Age effect is nonlinear** — charges rise sharply after 50, captured by `age_group` and `age_smoker` features
- **Linear models outperform tree ensembles** here — charge relationships are largely additive once interaction features are engineered
- **Charges are right-skewed** (skewness = 1.52) with 139 high-value outliers that are real data points, not errors

---

## Feature Engineering

8 features engineered on top of the 6 originals (7 → 15 columns, 20 after one-hot encoding):

| Feature        | Type        | Description                                                        |
| -------------- | ----------- | ------------------------------------------------------------------ |
| `age_group`    | Categorical | `young` / `middle_age` / `senior` — clinical age bands             |
| `bmi_category` | Categorical | `underweight` / `normal` / `overweight` / `obese` — WHO thresholds |
| `is_obese`     | Binary      | 1 if BMI ≥ 30                                                      |
| `smoker_obese` | Binary      | 1 if smoker AND obese — highest-risk interaction                   |
| `age_bmi`      | Numeric     | age × bmi / 1000 — compound metabolic risk                         |
| `has_children` | Binary      | 1 if at least one dependant                                        |
| `family_size`  | Categorical | `individual` / `small_family` / `large_family`                     |
| `age_smoker`   | Numeric     | age × smoker flag — older smokers pay far more                     |

---

## Outputs

After training, visualisations and reports are saved to `outputs/`:

| Output                    | Path                                                   |
| ------------------------- | ------------------------------------------------------ |
| Data understanding report | `outputs/reports/data_understanding.md`                |
| Model comparison report   | `outputs/reports/model_comparison.md`                  |
| Model metrics CSV         | `outputs/metrics/comparison.csv`                       |
| Cross-validation results  | `outputs/metrics/cross_validation.json`                |
| Learning curves           | `outputs/metrics/learning_curves.json`                 |
| Charges distribution      | `outputs/figures/eda/charges_distribution.png`         |
| Correlation heatmap       | `outputs/figures/eda/correlation_heatmap.png`          |
| Model comparison chart    | `outputs/figures/evaluation/model_comparison_rmse.png` |
| Residual plots            | `outputs/figures/evaluation/residuals_*.png`           |
| Feature importance        | `outputs/figures/evaluation/feature_importance_*.png`  |
| Learning curve plots      | `outputs/figures/evaluation/learning_curve_*.png`      |

---

## Bonus Features

- **5-Fold Cross-Validation** — CV R² and RMSE per model saved to `outputs/metrics/cross_validation.json`
- **Residual Plots** — per-model residual distribution and residuals vs predicted
- **Prediction vs Actual** — scatter plots for each model
- **Learning Curves** — train/validation R² vs training set size for key models
- **XGBoost & Extra Trees** — bonus models beyond the required 7

---

## Tech Stack

- Python 3.13
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-Learn, XGBoost, Joblib
- Streamlit
- Pytest

---

## License

MIT
