# Medical Insurance Cost Prediction — Project Guide

A concise step-by-step reference for building this end-to-end ML regression project from scratch.

---

## 1. Setup

- Create project folder structure (`data/`, `src/`, `models/`, `outputs/`, `app/`, `notebooks/`, `tests/`)
- Place dataset in `data/insurance.csv`
- Add `requirements.txt` and `.gitignore`
- Create virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate         # Mac / Linux
pip install -r requirements.txt
```

**Status: ✅ Done**

---

## 2. Understand the Data

- Load CSV with Pandas
- Inspect shape, columns, dtypes, missing values, duplicates
- Run descriptive statistics on all features
- Deep-dive the target (`charges`) — range, skewness, quartiles, outliers
- Compare smoker vs non-smoker charges
- Save reports to `outputs/reports/data_understanding.{md,json}`

**Key findings:**

- 1,338 rows · 7 columns · no missing values · 1 duplicate
- Charges range $1,122–$63,770 · skewness 1.52 (right-skewed)
- Smokers pay 3.8× more on average ($32,050 vs $8,434)
- 139 high-charge outliers — real data points, not errors

**Module:** `src/utils.py` → `generate_data_understanding_report()`

**Status: ✅ Done**

---

## 3. Clean the Data

- Remove 1 duplicate row
- Lowercase and strip all string columns
- Validate numeric ranges (age 0–120, BMI 10–80, children 0–10)
- Handle missing values — median for numeric, mode for categorical (defensive)
- Drop rows with missing target

**Module:** `src/preprocessing.py` → `clean_dataframe()`

**Status: ✅ Done**

---

## 4. Engineer Features

Create 8 new columns using only input features (no target leakage):

| Feature        | Description                                                |
| -------------- | ---------------------------------------------------------- |
| `age_group`    | young / middle_age / senior — clinical age bands           |
| `bmi_category` | underweight / normal / overweight / obese — WHO thresholds |
| `is_obese`     | 1 if BMI ≥ 30                                              |
| `smoker_obese` | 1 if smoker AND obese — highest-risk interaction           |
| `age_bmi`      | age × bmi / 1000 — compound metabolic risk                 |
| `has_children` | 1 if at least one dependant                                |
| `family_size`  | individual / small_family / large_family                   |
| `age_smoker`   | age × smoker flag — older smokers pay far more             |

**Module:** `src/feature_engineering.py` → `engineer_features()`

**Status: ✅ Done**

---

## 5. Build Preprocessing Pipeline

- Split columns into numeric (8) and categorical (6) after engineering
- Use `ColumnTransformer`:
  - **Numeric:** `SimpleImputer(median)` + `StandardScaler` (linear/SVR only)
  - **Categorical:** `SimpleImputer(most_frequent)` + `OneHotEncoder(drop="first")`
- Stratified 80/20 train/test split — stratified on `smoker × charge_quartile`
- Tree models skip scaling; linear and SVR models use it

**Module:** `src/preprocessing.py` → `build_preprocessor()` · `split_data()`

**Status: ✅ Done**

---

## 6. Exploratory Data Analysis

Generate and save plots to `outputs/figures/eda/`:

- Charges distribution (histogram + KDE)
- Charges by smoker status (boxplot)
- BMI distribution
- Age distribution
- Correlation heatmap
- Scatter: age vs charges, BMI vs charges coloured by smoker
- Pairplot of numeric features
- Charges by region (boxplot)
- Children vs average charges (bar)

Document findings in `notebooks/eda.ipynb`.

**Module:** `src/visualization.py` → `generate_all_eda_figures()`

**Status: ⏳ Next**

---

## 7. Train Models

Train 10 regressors inside sklearn `Pipeline` (preprocessor + estimator):

| Model                       | Scaling | Type     |
| --------------------------- | ------- | -------- |
| Linear Regression           | Yes     | Required |
| Ridge Regression            | Yes     | Required |
| Lasso Regression            | Yes     | Required |
| Decision Tree Regressor     | No      | Required |
| Random Forest Regressor     | No      | Required |
| Gradient Boosting Regressor | No      | Required |
| Support Vector Regressor    | Yes     | Required |
| XGBoost Regressor           | No      | Bonus    |
| Extra Trees Regressor       | No      | Bonus    |
| AdaBoost Regressor          | No      | Bonus    |

Split data 80/20 (stratified), fit each pipeline, record training time.

**Module:** `src/train.py` → `run_training_pipeline()`

**Status: ✅ Done**

---

## 8. Evaluate Models

For each model compute on the **original dollar scale**:

- MAE · MSE · RMSE · R²
- 5-fold cross-validation R² and RMSE
- Training time and prediction time

Save to `outputs/metrics/comparison.csv` and `cross_validation.json`.

**Module:** `src/evaluate.py`

**Status: ✅ Done**

---

## 9. Compare & Select Best Model

- Rank via composite score: RMSE 40% · MAE 35% · R² 15% · training time 10%
- Generate comparison table, bar chart, and leaderboard
- Save to `outputs/reports/model_comparison.md`

**Result:** Lasso Regression — RMSE $4,502 · MAE $2,438 · R² 0.8852 · CV R² 0.851

**Module:** `src/evaluate.py` → `select_best_model()`

**Status: ✅ Done**

---

## 10. Save Model

- Save best pipeline to `models/best_model.pkl` with Joblib
- Save feature metadata to `models/feature_info.json`
- Save model config to `models/model_metadata.json`

**Status: ✅ Done**

---

## 11. Evaluation Plots

Generate per-model visualisations to `outputs/figures/evaluation/`:

- Model comparison bar chart (RMSE)
- Model leaderboard (composite score)
- Residual distribution + residuals vs predicted (per model)
- Predicted vs actual scatter (per model)
- Feature importance (tree models)
- Coefficient plots (linear models)
- Learning curves (key models)

**Module:** `src/visualization.py`

**Status: ⏳ Next**

---

## 12. Build Inference Module

Load best model and run single + batch predictions:

```python
from src.predict import predict_charges

charges = predict_charges({
    "age": 35, "sex": "male", "bmi": 28.5,
    "children": 2, "smoker": "no", "region": "northwest"
})
print(f"${charges:,.2f}")
```

- Apply the same cleaning + feature engineering as training
- Clip predictions to ≥ 0
- Support both single dict and batch DataFrame

**Module:** `src/predict.py` → `predict_charges()` · `predict_batch()`

**Status: ⏳ Next**

---

## 13. Build Streamlit App

Create multi-page app in `app/app.py`:

| Page             | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| Home             | Project overview, best model metrics at a glance  |
| Dataset Overview | Shape, sample data, descriptive statistics        |
| EDA              | Display saved EDA figures                         |
| Model Comparison | Metrics table, comparison charts, markdown report |
| Predict Charges  | Input form → predicted insurance charge           |
| About            | Tech stack, project info                          |

Load saved model and feature info at runtime via `src/predict.py`.

**Module:** `app/app.py`

**Status: ⏳ Next**

---

## 14. EDA Notebook

Write `notebooks/eda.ipynb` covering:

- Data loading and inspection
- Target variable analysis (distribution, skewness, outliers)
- Feature relationships with charges
- Smoker vs non-smoker analysis
- Correlation analysis
- Engineered feature impact

**Status: ⏳ Next**

---

## 15. Unit Tests

Write `tests/test_preprocessing.py` covering:

- `remove_duplicates` removes correct number of rows
- `standardise_categoricals` lowercases and strips strings
- `validate_ranges` nullifies out-of-range values
- `clean_dataframe` output shape is correct
- `split_data` preserves smoker ratio
- `engineer_features` adds expected columns

```powershell
$env:PYTHONPATH = "."
pytest tests/ -v
```

**Status: ⏳ Next**

---

## Final Validation Checklist

Run these before considering the project complete:

```powershell
$env:PYTHONPATH = "."
python -m src.train                 # Full pipeline end-to-end
pytest tests/ -v                    # Unit tests
streamlit run app/app.py            # App launch
```

Verify:

- [ ] Preprocessing pipeline works on raw CSV
- [ ] All 10 models train and evaluate successfully
- [ ] Metrics and figures are generated to `outputs/`
- [ ] Best model saves and reloads correctly
- [ ] Predictions return valid dollar amounts
- [ ] Streamlit app runs without errors on all pages

---

## Quick Command Reference

| Task              | Command                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Inspect dataset   | `python -c "from src.utils import load_raw_data, generate_data_understanding_report; generate_data_understanding_report(load_raw_data())"` |
| Train all models  | `python -m src.train`                                                                                                                      |
| Run tests         | `pytest tests/ -v`                                                                                                                         |
| Launch app        | `streamlit run app/app.py`                                                                                                                 |
| Open EDA notebook | `jupyter notebook notebooks/eda.ipynb`                                                                                                     |

> Prefix all commands with `$env:PYTHONPATH = "."` on Windows PowerShell.

---

## Suggested Workflow Order

```
Setup → Data Understanding → Cleaning → Feature Engineering
    → Preprocessing Pipeline → Train Models → Evaluate
    → Compare & Select → Save Model → EDA Visualisations
    → Evaluation Plots → Inference Module → Streamlit App
    → EDA Notebook → Unit Tests → Final Validation
```

---

## Key Files Map

| File                         | Role                                                 |
| ---------------------------- | ---------------------------------------------------- |
| `src/utils.py`               | Paths, logger, data loading, inspection reports      |
| `src/preprocessing.py`       | Cleaning, validation, OHE pipeline, stratified split |
| `src/feature_engineering.py` | 8 domain-driven derived features                     |
| `src/train.py`               | End-to-end training orchestrator                     |
| `src/evaluate.py`            | Metrics, CV, learning curves, model selection        |
| `src/predict.py`             | Load model and run inference                         |
| `src/visualization.py`       | EDA + evaluation plots                               |
| `app/app.py`                 | Streamlit multi-page application                     |
| `notebooks/eda.ipynb`        | Interactive exploratory analysis                     |
| `models/best_model.pkl`      | Serialised best pipeline                             |
| `models/feature_info.json`   | Feature lists for inference layer                    |

---

_Follow these steps in order. Verify each phase before moving to the next._
