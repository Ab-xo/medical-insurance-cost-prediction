# 🏥 Medical Insurance Cost Prediction

An end-to-end machine learning regression project that predicts individual medical insurance charges based on demographic and health-related features. Built as part of the **Intelligent AI and Data Engineering** course group project.

> **Dataset:** [Medical Insurance Cost — Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset/data)
> **Target variable:** `charges` (USD)
> **Best model:** Lasso Regression — R² = 0.8852 | RMSE = $4,502 | MAE = $2,438

---

## 📌 Business Problem

Insurance companies need to price policies accurately. Underpricing leads to losses; overpricing drives customers away. This project builds a regression pipeline that predicts annual medical charges for a policyholder given their age, BMI, smoking status, number of children, sex, and region — enabling data-driven, fair pricing decisions.

---

## 📊 Dataset

| Feature    | Type        | Description                                                    |
| ---------- | ----------- | -------------------------------------------------------------- |
| `age`      | Numeric     | Age of the primary beneficiary                                 |
| `sex`      | Categorical | Gender — `male` / `female`                                     |
| `bmi`      | Numeric     | Body Mass Index (kg/m²)                                        |
| `children` | Numeric     | Number of dependants covered                                   |
| `smoker`   | Categorical | Smoking status — `yes` / `no`                                  |
| `region`   | Categorical | US region — `northeast`, `northwest`, `southeast`, `southwest` |
| `charges`  | **Target**  | Annual medical insurance cost (USD)                            |

**Key statistics from data understanding:**

| Stat                | Value                             |
| ------------------- | --------------------------------- |
| Rows                | 1,338 (1,337 after deduplication) |
| Missing values      | None                              |
| Duplicate rows      | 1 (removed)                       |
| Charges range       | $1,122 — $63,770                  |
| Charges mean        | $13,270                           |
| Charges skewness    | 1.52 (right-skewed)               |
| Smoker mean charges | $32,050 vs $8,434 (non-smoker)    |

---

## 🔧 Feature Engineering

8 domain-driven features added on top of the 6 original inputs:

| Feature        | Type        | Description                                                        |
| -------------- | ----------- | ------------------------------------------------------------------ |
| `age_group`    | Categorical | `young` / `middle_age` / `senior` (clinical age bands)             |
| `bmi_category` | Categorical | `underweight` / `normal` / `overweight` / `obese` (WHO thresholds) |
| `is_obese`     | Binary      | 1 if BMI ≥ 30                                                      |
| `smoker_obese` | Binary      | 1 if smoker AND obese — highest-risk interaction                   |
| `age_bmi`      | Numeric     | `age × bmi / 1000` — compound metabolic risk                       |
| `has_children` | Binary      | 1 if at least one dependant                                        |
| `family_size`  | Categorical | `individual` / `small_family` / `large_family`                     |
| `age_smoker`   | Numeric     | `age × smoker_flag` — older smokers pay far more                   |

---

## 🤖 Models Compared

<cite index="1-12,1-13">The project compares **10 regression algorithms** (7 required + 3 bonus):</cite>

| Rank | Model                           | RMSE   | MAE    | R²     | CV R² | Train Time |
| ---- | ------------------------------- | ------ | ------ | ------ | ----- | ---------- |
| ⭐ 1 | **Lasso Regression**            | $4,502 | $2,438 | 0.8852 | 0.851 | 0.08s      |
| 2    | Linear Regression               | $4,503 | $2,439 | 0.8852 | 0.851 | 0.03s      |
| 3    | Ridge Regression                | $4,552 | $2,545 | 0.8827 | 0.848 | 0.10s      |
| 4    | XGBoost Regressor _(bonus)_     | $4,703 | $2,456 | 0.8747 | 0.842 | 0.86s      |
| 5    | Gradient Boosting Regressor     | $4,743 | $2,451 | 0.8726 | 0.838 | 1.42s      |
| 6    | Random Forest Regressor         | $4,823 | $2,617 | 0.8683 | 0.825 | 1.05s      |
| 7    | Extra Trees Regressor _(bonus)_ | $4,963 | $2,693 | 0.8605 | 0.813 | 0.60s      |
| 8    | Support Vector Regressor        | $5,100 | $2,000 | 0.8527 | 0.811 | 0.32s      |
| 9    | AdaBoost Regressor _(bonus)_    | $5,723 | $4,866 | 0.8145 | 0.787 | 1.32s      |
| 10   | Decision Tree Regressor         | $6,396 | $3,207 | 0.7683 | 0.744 | 0.10s      |

<cite index="1-14">**Evaluation metrics used:** MAE, MSE, RMSE, R²</cite> — plus 5-fold cross-validation R² for each model.

**Why Lasso?** Selected via composite ranking (RMSE 40%, MAE 35%, R² 15%, training time 10%). Lasso's L1 regularisation naturally shrinks irrelevant feature coefficients to zero, acting as built-in feature selection — well-suited to this dataset where smoker status dominates and many other features have smaller marginal effects.

---

## 🏗️ Project Structure

```
medical-insurance-cost-prediction/
│
├── data/
│   └── insurance.csv                  # Raw dataset (1,338 rows)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Paths, logger, data inspection, I/O
│   ├── preprocessing.py               # Cleaning, validation, split
│   ├── feature_engineering.py         # 8 domain-driven engineered features
│   ├── evaluate.py                    # Metrics, CV, learning curves, selection
│   └── train.py                       # Full training pipeline (10 models)
│
├── models/
│   ├── best_model.pkl                 # Saved best model pipeline (generated)
│   ├── feature_info.json              # Feature lists for inference
│   └── model_metadata.json            # Best model name + config
│
├── outputs/
│   ├── figures/
│   │   ├── eda/                       # EDA visualisations (generated)
│   │   └── evaluation/                # Model eval plots (generated)
│   ├── metrics/
│   │   ├── comparison.csv             # All model metrics (generated)
│   │   ├── cross_validation.json      # CV results (generated)
│   │   └── learning_curves.json       # Learning curve data (generated)
│   └── reports/
│       ├── data_understanding.md      # Auto-generated data report
│       ├── model_comparison.md        # Auto-generated model report
│       └── cleaned_data.csv           # Enriched dataset for app (generated)
│
├── app/
│   └── app.py                         # Streamlit web application (coming)
│
├── notebooks/
│   └── eda.ipynb                      # Exploratory analysis notebook (coming)
│
├── tests/
│   └── test_preprocessing.py          # Unit tests (coming)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.11+ (tested on 3.13)
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Ab-xo/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
```

### 2. Create and activate virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run data inspection

```powershell
# Windows PowerShell
$env:PYTHONPATH = "."
python -c "from src.utils import load_raw_data, generate_data_understanding_report; generate_data_understanding_report(load_raw_data())"
```

### Train all models

```powershell
$env:PYTHONPATH = "."
python -m src.train
```

This runs the full pipeline:

1. Loads and cleans `data/insurance.csv`
2. Engineers 8 new features
3. Splits 80 / 20 (stratified by smoker × charge quartile)
4. Trains and evaluates 10 regression models
5. Runs 5-fold cross-validation on each
6. Saves the best model to `models/best_model.pkl`
7. Writes comparison reports to `outputs/`

### Launch the web app _(coming soon)_

```bash
streamlit run app/app.py
```

---

## 📦 Dependencies

| Package        | Version | Purpose                         |
| -------------- | ------- | ------------------------------- |
| `pandas`       | ≥ 2.1   | Data manipulation               |
| `numpy`        | ≥ 1.26  | Numerical computing             |
| `scikit-learn` | ≥ 1.4   | ML models & pipelines           |
| `xgboost`      | ≥ 2.0   | Gradient boosting (bonus model) |
| `matplotlib`   | ≥ 3.8   | Visualisation                   |
| `seaborn`      | ≥ 0.13  | Statistical plots               |
| `plotly`       | ≥ 5.20  | Interactive charts              |
| `streamlit`    | ≥ 1.33  | Web application                 |
| `joblib`       | ≥ 1.3   | Model serialisation             |
| `jupyter`      | ≥ 1.0   | Notebooks                       |
| `pytest`       | ≥ 8.0   | Unit tests                      |

---

## 📈 Key Findings

- **Smoker status** is the single strongest predictor — smokers pay **3.8× more** on average
- **BMI ≥ 30 + smoking** creates the highest-cost group (compounding interaction)
- **Age** has a nonlinear effect — costs rise sharply after 50
- **Charges are right-skewed** (skewness = 1.52) — linear models still perform well after L1 regularisation
- Linear models (Lasso, Linear, Ridge) **outperform tree ensembles** on this dataset — the relationships are largely linear once `smoker_obese` and `age_smoker` interactions are engineered

---

## 🗺️ Roadmap

- [x] Data understanding & inspection
- [x] Data cleaning & preprocessing
- [x] Feature engineering (8 features)
- [x] Model training — 10 algorithms
- [x] Cross-validation & evaluation
- [ ] EDA visualisations (`visualization.py`)
- [ ] Inference module (`predict.py`)
- [ ] Streamlit web application (`app/app.py`)
- [ ] EDA Jupyter notebook (`notebooks/eda.ipynb`)
- [ ] Unit tests (`tests/`)
- [ ] Deployment

---

## 👥 Group 3 — Course Project

<cite index="1-25">This project is assigned to **Group 3: Medical Insurance Cost** with target variable **Insurance Charges**.</cite>

<cite index="1-1">**Course:** Intelligent AI and Data Engineering</cite>

---

## 📄 License

This project is for educational purposes as part of a course group assignment.
