# Medical Insurance Cost Prediction

An end-to-end machine-learning application that predicts annual medical insurance charges from policyholder attributes. The project combines a Python modeling pipeline, FastAPI inference service, and Next.js analytics dashboard.

## Highlights

- Compares ten regression models through a consistent sklearn pipeline
- Uses feature engineering for age, BMI, smoking, and family interactions
- Selects a model using cross-validated performance
- Serves live predictions through a FastAPI REST API
- Presents metrics, EDA, model comparisons, and predictions in a dashboard
- Includes automated tests for preprocessing, inference, and API behavior

## Technology

| Layer | Stack |
|---|---|
| ML pipeline | Python, pandas, NumPy, scikit-learn, XGBoost, joblib |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Next.js, React, TypeScript, Tailwind CSS, Recharts |
| Testing | pytest |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cd frontend && npm ci && cd ..
```

```bash
PYTHONPATH=. python -m src.train
uvicorn main:app --reload --port 8000
cd frontend && npm run dev
```

## Testing

```bash
PYTHONPATH=. pytest tests/ -v
```

## Disclaimer

Predictions are educational estimates and should not be used as financial, medical, or insurance advice.

## License

MIT
