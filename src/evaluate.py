# src/evaluate.py

import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from .config import DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true, probas, threshold=DEFAULT_THRESHOLD):
    """
    Evaluate probability predictions using multiple metrics.
    """

    preds = (probas >= threshold).astype(int)

    results = {
        "roc_auc": roc_auc_score(y_true, probas),
        "pr_auc": average_precision_score(y_true, probas),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "threshold": threshold
    }

    return results


def print_eval_results(name, results: dict):
    logger.info(f"\n===== {name.upper()} EVALUATION =====")
    for k, v in results.items():
        logger.info(f"{k}: {v:.5f}")
