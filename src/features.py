import pandas as pd
import numpy as np
from .config import TARGET_COL


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Adds engineered feature column to dataframe.
        Returns a new dataframe ( does not mutate original).
    """
    df = df.copy()

    # Cardiovascular Features
    if { "systolic_bp", "diastolic_bp"}.issubset(df.columns):
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        df["pulse_pressure_ratio"] = df["pulse_pressure"]/ (df["systolic_bp"] + 1e-6)
        df["mean_arterial_pressure"] = (
            (df["systolic_bp"] + 2*df["diastolic_bp"])/3
        )
    else:
        df["pulse_pressure"] = 0
        df["pulse_pressure_ratio"] = 0
        df["mean_arterial_pressure"] = 0

    if "heart_rate" in df.columns and "systolic_bp" in df.columns:
        df['rate_pressure_product'] = df["heart_rate"]*df["systolic_bp"]
    else:
        df["rate_pressure_product"] = 0

    #Lipid Profile Features
    lipid_cols = {"cholesterol","ldl","hdl", "triglycerides"}
    if lipid_cols.issubset(df.columns):
        df["ldl_hdl_ratio"] = df["ldl"] / (df["hdl"] + 1e-6)
        df["chol_hdl_ratio"] = df["cholesterol"] / (df["hdl"] + 1e-6)
        df["non_hdl_cholesterol"] = df["cholesterol"] - df["hdl"]
        df["ldl_share"] = df["ldl"] / (df["cholesterol"] + 1e-6)
        df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl"] + 1e-6)
        df["lipid_sum"] = df["cholesterol"] + df["triglycerides"]
        df["lipid_burden"] = (
            df["ldl_hdl_ratio"] + df["tg_hdl_ratio"] + df["chol_hdl_ratio"]
        )
    else:
        df["ldl_hdl_ratio"] = 0
        df["chol_hdl_ratio"] = 0
        df["non_hdl_cholesterol"] = 0
        df["ldl_share"] = 0
        df["tg_hdl_ratio"] = 0
        df["lipid_sum"] = 0
        df["lipid_burden"] = 0
        
    # Lifestyle Features
    if "age" in df.columns and "bmi" in df.columns:
        df["age_bmi_risk"] = df["age"] * df["bmi"]
    else:
        df["age_bmi_risk"] = 0

    if "age" in df.columns and "physical_activity" in df.columns:
        df["activity_age_ratio"] = df["physical_activity"] / (df["age"] + 1e-6)
        df["activity_x_age"] = df["physical_activity"] * df["age"]
    else:
        df["activity_age_ratio"] = 0
        df["activity_x_age"] = 0

    if {"screen_time", "physical_activity"}.issubset(df.columns):
        df["screen_activity_ratio"] = df["screen_time"] / (
            df["physical_activity"] + 1e-6
        )
    else:
        df["screen_activity_ratio"] = 0

    # lifestyle composite score (safe defaults if missing)
    df["lifestyle_risk_score"] = (
        0.3 * df.get("bmi", 0)
        + 0.2 * df.get("waist_to_hip_ratio", 0)
        + 0.2 * df.get("screen_time", 0)
        - 0.2 * df.get("physical_activity", 0)
        - 0.1 * df.get("sleep_duration", 0)
    )

    # History / Risk Combinations

    df["risk_history"] = (
        df.get("hypertension_history", 0) + df.get("cardiovascular_history", 0)
    )

    if "family_history" in df.columns and "bmi" in df.columns:
        df["genetic_history_risk"] = df["family_history"] * df["bmi"]
    else:
        df["genetic_history_risk"] = 0

    if "age" in df.columns and "mean_arterial_pressure" in df.columns:
        df["age_map_risk"] = df["age"] * df["mean_arterial_pressure"]
    else:
        df["age_map_risk"] = 0

    # Cleanups

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

def get_feature_columns(df: pd.DataFrame):
    """
        Return Feature column list excluding ID and target
    """

    drop_cols = set()

    # common Id Columns

    for col in ["id", "ID", "patient_id"]:
        if col in df.columns:
            drop_cols.add(col)
        
    #target
    if TARGET_COL in df.columns:
        drop_cols.add(TARGET_COL)

    return [c for c in df.columns if c not in drop_cols]


def build_train_matrix(df: pd.DataFrame):
    """
    Applies feature engineering and returns X, y
    """
    df = create_features(df)
    features = get_feature_columns(df)
    X =df[features]
    y = df[TARGET_COL]

    return X, y, features

def build_test_matrix(df: pd.DataFrame):
    """
        Applies feature engineering and returns X for test data
    """
    df = create_features(df)
    features = get_feature_columns(df)
    X = df[features]

    return X, features

