# Medical Insurance Cost Prediction

End-to-end machine learning pipeline that predicts annual US medical insurance
charges from six policyholder attributes, served through a **Next.js dashboard**
and a **FastAPI REST backend**.

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Project Does

Given a policyholder's age, sex, BMI, number of dependants, smoking status, and
US region — the model predicts their **annual insurance cost in USD**.

- Trains and compares **10 regression models** using a full sklearn pipeline
- Selects the best model using **cross-validated composite ranking**
- Serves predictions live via a **FastAPI JSON API**
- Visualises everything through a **Next.js dashboard** (plots, metrics, live form)

---

## Live Results

| Rank | Model                             |       RMSE |        MAE |         R² |     CV R² |
| ---: | --------------------------------- | ---------: | ---------: | ---------: | --------: |
| ⭐ 1 | **Lasso Regression** (CV-tuned α) | **$4,493** | **$2,433** | **0.8857** | **0.851** |
|    2 | Linear Regression                 |     $4,503 |     $2,439 |     0.8852 |     0.851 |
|    3 | Ridge Regression (CV-tuned α)     |     $4,504 |     $2,443 |     0.8851 |     0.848 |
|    4 | Extra Trees Regressor             |     $4,573 |     $2,454 |     0.8816 |     0.842 |
|    5 | Support Vector Regressor          |     $4,610 |     $1,490 |     0.8797 |     0.841 |
|    6 | Decision Tree Regressor           |     $4,576 |     $2,533 |     0.8814 |     0.835 |
|    7 | Random Forest Regressor           |     $4,588 |     $2,470 |     0.8808 |     0.844 |
|    8 | Gradient Boosting Regressor       |     $4,708 |     $2,467 |     0.8745 |     0.838 |
|    9 | AdaBoost Regressor                |     $5,448 |     $4,402 |     0.8319 |     0.790 |
|   10 | XGBoost Regressor                 |     $4,659 |     $2,407 |     0.8771 |     0.844 |

**Alpha selection:** `LassoCV` and `RidgeCV` search the optimal regularisation
strength automatically via 5-fold cross-validation — no hardcoded guesses.

---

## Key Findings

- **Smoker status** is the dominant predictor — smokers pay **3.8× more** ($32,050 vs $8,434)
- **BMI ≥ 30 + smoking** creates the highest-charge group, captured by the engineered `smoker_obese` interaction
- **Age effect is nonlinear** — charges spike sharply after 50 (`age_group`, `age_smoker` features)
- **Linear models beat tree ensembles** once interaction features are engineered
- Charges are **right-skewed** (skewness 1.52) with 139 real high-value outliers

---

## Dataset

| Column     | Type  | Description                                           |
| ---------- | ----- | ----------------------------------------------------- |
| `age`      | int   | Age of the primary beneficiary (18–64)                |
| `sex`      | str   | `male` / `female`                                     |
| `bmi`      | float | Body Mass Index (kg/m²)                               |
| `children` | int   | Number of dependants (0–5)                            |
| `smoker`   | str   | `yes` / `no`                                          |
| `region`   | str   | `northeast` / `northwest` / `southeast` / `southwest` |
| `charges`  | float | Annual insurance cost in USD — **target**             |

- **1,338 policyholders** (1,337 after deduplication)
- **No missing values**
- Source: [Kaggle — Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset/data)

---

## Prerequisites

| Tool    | Version | Notes                 |
| ------- | ------- | --------------------- |
| Python  | 3.11+   | Tested on 3.13        |
| Node.js | 18+     | For Next.js dashboard |
| npm     | 9+      | Bundled with Node.js  |
| Git     | any     | To clone the repo     |

---

## Installation & Setup

### 1 — Clone the repository

```bash
git clone https://github.com/Ab-xo/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
```

### 2 — Python environment

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3 — Node environment (first time only)

```bash
cd frontend
npm install
cd ..
```

---

## Running the Project

All three steps below can run simultaneously in separate terminals.

### Step 1 — Train the models

```powershell
# Windows
$env:PYTHONPATH = "."
python -m src.train
```

```bash
# macOS / Linux
PYTHONPATH=. python -m src.train
```

This takes ~2 minutes and produces:

- `models/best_model.pkl` — serialised winning pipeline
- `outputs/figures/` — 12 EDA + 36 evaluation plots
- `outputs/metrics/` — comparison CSV, cross-validation JSON
- `outputs/reports/` — markdown comparison report

> Skip this step if `models/best_model.pkl` already exists.

### Step 2 — Start the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### Step 3 — Start the Next.js dashboard

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`

---

## Dashboard Pages

| Page                 | Route      | What you see                                                           |
| -------------------- | ---------- | ---------------------------------------------------------------------- |
| **Home**             | `/`        | KPI cards, regional averages, model leaderboard, 8 engineered features |
| **Dataset**          | `/dataset` | Shape, column types, first 20 rows, descriptive statistics             |
| **EDA**              | `/eda`     | 12 plots — tabbed by category, each with fullscreen modal              |
| **Model Comparison** | `/metrics` | Leaderboard, RMSE/R² charts, per-model residuals & feature importance  |
| **Predict**          | `/predict` | Live form → POST to FastAPI → prediction card with risk flags          |
| **About**            | `/about`   | Project details, tech stack, quick commands                            |

---

## API Endpoints

Base URL: `http://localhost:8000`

| Method | Endpoint                   | Description                                   |
| ------ | -------------------------- | --------------------------------------------- |
| `POST` | `/api/predict`             | Predict charges for a policyholder            |
| `GET`  | `/api/metrics`             | All 10 model scores sorted by RMSE            |
| `GET`  | `/api/cv`                  | 5-fold cross-validation results               |
| `GET`  | `/api/dataset/summary`     | Shape, target stats, smoker/region breakdowns |
| `GET`  | `/api/dataset/sample`      | First 20 rows as JSON                         |
| `GET`  | `/api/feature-info`        | Best model name + feature lists               |
| `GET`  | `/api/eda-figures`         | List of EDA figure URLs                       |
| `GET`  | `/api/eval-figures/{stem}` | Per-model evaluation plot URLs                |
| `GET`  | `/docs`                    | Interactive Swagger UI                        |

**Example prediction request:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"male","bmi":28.5,"children":2,"smoker":"no","region":"northwest"}'
```

**Response:**

```json
{
  "predicted_charges": 7368.0,
  "formatted": "$7,368.00",
  "model_used": "Lasso Regression",
  "risk_flags": [],
  "context": {
    "overall_mean": 13279.12,
    "smoker_mean": 32050.23,
    "nonsmoker_mean": 8434.27
  }
}
```

---

## Running Tests

```powershell
# Windows
$env:PYTHONPATH = "."
pytest tests/ -v
```

```bash
# macOS / Linux
PYTHONPATH=. pytest tests/ -v
```

**52 tests** cover preprocessing, feature engineering, inference, and the FastAPI endpoints.

---

## Project Structure

```
medical-insurance-cost-prediction/
│
├── frontend/                    # Next.js 16 dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Home
│   │   │   ├── dataset/         # Dataset overview
│   │   │   ├── eda/             # EDA plots
│   │   │   ├── metrics/         # Model comparison
│   │   │   ├── predict/         # Live prediction form
│   │   │   └── about/           # Project info
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── ChartCard.tsx    # Minimise / maximise
│   │   │   ├── ImageModal.tsx   # Fullscreen with prev/next
│   │   │   └── ui/              # shadcn/ui primitives
│   │   └── lib/utils.ts
│   ├── next.config.ts
│   ├── package.json
│   └── tsconfig.json
│
├── src/                         # Python ML library
│   ├── utils.py                 # Paths, logger, helpers
│   ├── preprocessing.py         # Clean, validate, split
│   ├── feature_engineering.py   # 8 engineered features
│   ├── train.py                 # Train 10 models (LassoCV / RidgeCV)
│   ├── evaluate.py              # Metrics, CV, model selection
│   ├── predict.py               # Single & batch inference
│   └── visualization.py         # EDA + evaluation plots
│
├── data/
│   └── insurance.csv            # Raw dataset (1,338 rows)
│
├── models/                      # Generated by training
│   ├── best_model.pkl           # Winning pipeline (joblib)
│   ├── feature_info.json        # Feature metadata
│   └── model_metadata.json
│
├── outputs/                     # Generated by training
│   ├── figures/eda/             # 12 EDA plots
│   ├── figures/evaluation/      # 36 model evaluation plots
│   ├── metrics/                 # CSVs and JSON results
│   └── reports/                 # Markdown reports
│
├── notebooks/
│   └── eda.ipynb                # Exploratory notebook
│
├── tests/                       # 52 pytest tests
│   ├── test_preprocessing.py
│   ├── test_predict.py
│   └── test_api.py
│
├── main.py                      # FastAPI backend (port 8000)
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## Feature Engineering

8 features added on top of the 6 raw inputs:

| Feature        | Type        | Description                                      |
| -------------- | ----------- | ------------------------------------------------ |
| `age_group`    | categorical | young / middle_age / senior                      |
| `bmi_category` | categorical | underweight / normal / overweight / obese        |
| `is_obese`     | binary      | 1 if BMI ≥ 30                                    |
| `smoker_obese` | binary      | 1 if smoker AND obese — highest-risk interaction |
| `age_bmi`      | numeric     | age × bmi / 1000 — compound metabolic risk       |
| `has_children` | binary      | 1 if dependants > 0                              |
| `family_size`  | categorical | individual / small_family / large_family         |
| `age_smoker`   | numeric     | age × smoker flag — older smokers pay far more   |

---

## Tech Stack

| Layer             | Technology                                                                            |
| ----------------- | ------------------------------------------------------------------------------------- |
| Next.js dashboard | Next.js 16, React 19, TypeScript, Tailwind CSS v4, shadcn/ui, Recharts, Framer Motion |
| REST API          | FastAPI, Uvicorn, Pydantic v2                                                         |
| ML pipeline       | scikit-learn, XGBoost, pandas, numpy, joblib                                          |
| Visualisation     | matplotlib, seaborn                                                                   |
| Testing           | pytest                                                                                |
| Language          | Python 3.13, TypeScript 5                                                             |

---

## License

MIT — see [LICENSE](LICENSE) for details.
