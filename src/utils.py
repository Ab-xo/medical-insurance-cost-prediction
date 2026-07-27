from pathlib import Path
import pandas as pd
import json
import logging

# ROOT_DIR = the project's top folder,computed automatically

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "insurance.csv"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

TARGET_COLUMN = "charges"
RANDOM_STATE = 42   # fixed seed so results are reproducible every run


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle and place it at data/insurance.csv"
        )
    return pd.read_csv(path)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def inspect_dataframe(df: pd.DataFrame, logger=None) -> None:
    log = logger or get_logger("inspect")
    log.info("Shape: %s", df.shape)
    log.info("Missing values:\n%s", df.isna().sum())
    log.info("Duplicate rows: %d", df.duplicated().sum())
    log.info("Numeric summary:\n%s", df.describe().T)
