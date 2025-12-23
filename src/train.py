import os
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from .config import (
    TRAIN_FILE,
    MODEL_DIR,
    CV_FOLDS,
    SEED,
    SHUFFLE,
    TARGET_COL,
)
from .features import build_train_matrix
from .models import get_all_models

logger = logging.getLogger(__name__)


def save_fold_model(model_name, fold, model):
    path = MODEL_DIR / model_name
    os.makedirs(path, exist_ok=True)

    model_path = path / f"{model_name}_fold{fold}.pkl"
    joblib.dump(model, model_path)


def cross_validate_model(model_name, model, X, y):
    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=SHUFFLE,
        random_state=SEED
    )

    oof_preds = np.zeros(len(X))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"\n===== {model_name.upper()} | FOLD {fold} =====")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred

        fold_score = roc_auc_score(y_val, val_pred)
        scores.append(fold_score)

        logger.info(f"Fold {fold} ROC-AUC: {fold_score:.5f}")

        save_fold_model(model_name, fold, model)

    logger.info(f"\n{model_name.upper()} CV Mean ROC-AUC: {np.mean(scores):.5f}")
    logger.info(f"{model_name.upper()} CV Std: {np.std(scores):.5f}")

    return oof_preds, scores


def run_training():
    logger.info("Loading processed training data...")
    df = pd.read_csv(TRAIN_FILE)

    logger.info("Building training matrix with features...")
    X, y, features = build_train_matrix(df)

    models = get_all_models()

    all_oof = {}
    all_scores = {}

    for name, model in models.items():
        oof_preds, scores = cross_validate_model(name, model, X, y)
        all_oof[name] = oof_preds
        all_scores[name] = scores

    logger.info("\nTraining completed successfully.")
    return all_oof, all_scores
