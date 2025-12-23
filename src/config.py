import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
#  DATA PATHS

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# ARTIFACT PATHS
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
LOG_DIR = ARTIFACTS_DIR / "logs"
SUBMISSION_DIR = ARTIFACTS_DIR / "submissions"

# Ensure directories exist
for path in [PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, SUBMISSION_DIR]:
    os.makedirs(path, exist_ok=True)


# General Setting
SEED  = 42
TARGET_COL = "diagnosed_diabetes"

# Cross Validation Settings
CV_FOLDS = 5
CV_STRATIFIED = True
SHUFFLE = True

# Model Names
MODEL_NAMES = {
    "lgbm":"lightgbm",
    "xgb":"xgboost",
    "cat":"catboost"
}

#Threshold Settings
DEFAULT_THRESHOLD = 0.55 

# Ensemble Weight
ENSEMBLE_WEIGHTS = {
    "lightgbm": 0.25,
    "xgboost": 0.30,
    "catboost": 0.45
}

# MODEL DEFAULT PARAMS

LIGHTGBM_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "objective": "binary"
}

XGBOOST_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": SEED,
    "tree_method": "hist"  # GPU users may switch to "gpu_hist"
}

CATBOOST_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "random_seed": SEED
}