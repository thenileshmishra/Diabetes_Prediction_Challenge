import os
import logging
import numpy as np
import pandas as pd
import joblib
from src.exception import CustomException
import sys

from .config import (
    MODEL_DIR,
    SUBMISSION_DIR,
    TEST_FILE,
    ENSEMBLE_WEIGHTS,
)

from .features import build_test_matrix

logger = logging.getLogger(__name__)

def load_models_for_type(model_name):
    path = MODEL_DIR / model_name
    models = []

    for file in sorted(os.listdir(path)):
        if file.endswith(".pkl"):
            models.append(joblib.load(path / file))

    if not models:
        raise CustomException(f"No saved models found for {model_name}")
    
    return models

def predict_with_models(models, X):
    preds = []

    for m in models:
        p = m.predict_proba(X)[:,1]
        preds.append(p)
    return np.mean(preds, axis=0)

def run_ensemble():
    logger.info("Loading processed test data...")
    df_test = pd.read_csv(TEST_FILE)

    logger.info("Building test matrix...")
    X_test, features = build_test_matrix(df_test)

    final_pred = np.zeros(len(X_test))

    for model_name, weight in ENSEMBLE_WEIGHTS.items():
        logger.info(f"Using {model_name} with weight {weight}")

        models = load_models_for_type(model_name)
        probas = predict_with_models(models, X_test)

        final_pred += weight * probas

    submission = pd.DataFrame({
        "id": df_test["id"],
        "diagnosed_diabetes": final_pred
    })

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(save_path, index=False)

    logger.info(f"Submission saved -> {save_path}")
    return submission