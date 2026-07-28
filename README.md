# 🏥 Medical Insurance Cost Prediction

An end-to-end machine learning regression project that predicts individual medical insurance charges from demographic and health features.

> **Dataset:** [Medical Insurance Cost — Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset/data) · 1,338 rows · 6 features · target: `charges` (USD)
> **Best model:** Lasso Regression — R² `0.8852` | RMSE `$4,502` | MAE `$2,438`

---

## Dataset

| Feature    | Type        | Description                                     |
| ---------- | ----------- | ----------------------------------------------- |
| `age`      | Numeric     | Age of the primary beneficiary                  |
| `sex`      | Categorical | `male` / `female`                               |
| `bmi`      | Numeric     | Body Mass Index (kg/m²)                         |
| `children` | Numeric     | Number of dependants covered                    |
| `smoker`   | Categorical | `yes` / `no`                                    |
| `region`   | Categorical | `northeast` `northwest` `southeast` `southwest` |
| `charges`  | **Target**  | Annual medical insurance cost (USD)             |

**Quick stats:** no missing values · 1 duplicate removed · charges range $1,122–$63,770 · skewness 1.52 · smokers pay $32,050 vs $8,434 average (non-smokers)

---

## Feature Engineering

8 domain-driven features added on top of the 6 originals:

| Feature        | Type        | Reasoning                                                  |
| -------------- | ----------- | ---------------------------------------------------------- |
| `age_group`    | Categorical | young / middle_age / senior — clinical age bands           |
| `bmi_category` | Categorical | underweight / normal / overweight / obese — WHO thresholds |
| `is_obese`     | Binary      | BMI ≥ 30                                                   |
| `smoker_obese` | Binary      | smoker AND obese — highest-risk compound group             |
| `age_bmi`      | Numeric     | age × bmi / 1000 — compound metabolic risk                 |
| `has_children` | Binary      | at least one dependant                                     |
| `family_size`  | Categorical | individual / small_family / large_family                   |
| `age_smoker`   | Numeric     | age × smoker flag — older smokers pay far more             |

---

## Model Comparison

10 regression algorithms trained and evaluated (7 required + 3 bonus):

| Rank | Model                    | RMSE   | MAE    | R²     | CV R² | Time  |
| ---- | ------------------------ | ------ | ------ | ------ | ----- | ----- |
| ⭐ 1 | **Lasso Regression**     | $4,502 | $2,438 | 0.8852 | 0.851 | 0.08s |
| 2    | Linear Regression        | $4,503 | $2,439 | 0.8852 | 0.851 | 0.03s |
| 3    | Ridge Regression         | $4,552 | $2,545 | 0.8827 | 0.848 | 0.10s |
| 4    | XGBoost _(bonus)_        | $4,703 | $2,456 | 0.8747 | 0.842 | 0.86s |
| 5    | Gradient Boosting        | $4,743 | $2,451 | 0.8726 | 0.838 | 1.42s |
| 6    | Random Forest            | $4,823 | $2,617 | 0.8683 | 0.825 | 1.05s |
| 7    | Extra Trees _(bonus)_    | $4,963 | $2,693 | 0.8605 | 0.813 | 0.60s |
| 8    | Support Vector Regressor | $5,100 | $2,000 | 0.8527 | 0.811 | 0.32s |
| 9    | AdaBoost _(bonus)_       | $5,723 | $4,866 | 0.8145 | 0.787 | 1.32s |
| 10   | Decision Tree            | $6,396 | $3,207 | 0.7683 | 0.744 | 0.10s |

Metrics: MAE · MSE · RMSE · R² · 5-fold cross-validation R² per model.

**Why Lasso?** Selected via composite rank (RMSE 40% · MAE 35% · R² 15% · time 10%). L1 regularisation shrinks irrelevant coefficients to zero — effective built-in feature selection when smoker status dominates the signal.

---

## Project Structure

```
medical-insurance-cost-prediction/
├── data/
│   └── insurance.csv                 # Raw dataset
├── src/
│   ├── utils.py                      # Paths, logger, data inspection, I/O
│   ├── preprocessing.py              # Cleaning, validation, stratified split
│   ├── feature_engineering.py        # 8 engineered features
│   ├── evaluate.py                   # Metrics, CV, learning curves, selection
│   └── train.py                      # Full pipeline — 10 models
├── models/
│   ├── best_model.pkl                # Saved pipeline (generated)
│   ├── feature_info.json
│   └── model_metadata.json
├── outputs/
│   ├── figures/eda/                  # EDA plots (generated)
│   ├── figures/evaluation/           # Model eval plots (generated)
│   ├── metrics/                      # comparison.csv, CV results
│   └── reports/                      # Markdown reports, cleaned_data.csv
├── app/
│   └── app.py                        # Streamlit app (coming)
├── notebooks/
│   └── eda.ipynb                     # EDA notebook (coming)
├── tests/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Ab-xo/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
python -m venv .venv
```

**Windows:** `.\.venv\Scripts\Activate.ps1`
**Mac/Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

---

## Usage

**Inspect the dataset:**

```powershell
$env:PYTHONPATH = "."
python -c "from src.utils import load_raw_data, generate_data_understanding_report; generate_data_understanding_report(load_raw_data())"
```

**Train all models:**

```powershell
$env:PYTHONPATH = "."
python -m src.train
```

Outputs: `models/best_model.pkl` · `outputs/metrics/comparison.csv` · `outputs/reports/model_comparison.md`

**Launch web app** _(coming soon)_:

```bash
streamlit run app/app.py
```

---

## Key Findings

- Smoker status is the dominant predictor — smokers pay **3.8× more** on average
- BMI ≥ 30 combined with smoking creates the highest-charge group
- Age effect is nonlinear — costs rise sharply after 50
- Linear models (Lasso, Ridge) outperform tree ensembles here — the charge relationships are largely additive once interaction features are engineered

---

## Roadmap

- [x] Data understanding & inspection
- [x] Cleaning & preprocessing
- [x] Feature engineering
- [x] Training & evaluation — 10 models
- [x] Cross-validation
- [ ] EDA visualisations
- [ ] Inference module (`predict.py`)
- [ ] Streamlit web app
- [ ] EDA notebook
- [ ] Unit tests
- [ ] Deployment
