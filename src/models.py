from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from .config import (
    LIGHTGBM_PARAMS,
    XGBOOST_PARAMS,
    CATBOOST_PARAMS,
    MODEL_NAMES,
)


def get_lightgbm():
    return LGBMClassifier(**LIGHTGBM_PARAMS)

def get_xgboost():
    return XGBClassifier(**XGBOOST_PARAMS)

def get_catboost():
    return CatBoostClassifier(**CATBOOST_PARAMS)


def get_all_models():
    """
    Returns dictionary of initalized models.
    """
    return {
        MODEL_NAMES["lgbm"]:get_lightgbm(),
        MODEL_NAMES["xgb"]:get_xgboost(),
        MODEL_NAMES["cat"]:get_catboost(),
    }