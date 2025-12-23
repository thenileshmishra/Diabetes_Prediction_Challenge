import pandas as pd
import logging
from pathlib import Path
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_FILE, TEST_FILE
from src.exception import CustomException
import sys

logger = logging.getLogger(__name__)


def load_raw_data():
    try:
        train_path = RAW_DATA_DIR / "train.csv"
        test_path = RAW_DATA_DIR / "test.csv"

        if not train_path.exists():
            raise CustomException(f"Missing file: {train_path}")

        if not test_path.exists():
            raise CustomException(f"Missing file: {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        return train_df, test_df

    except Exception as e:
        raise CustomException(e, sys)


def basic_quality_check(train_df, test_df):
    try:
        if train_df.empty:
            raise CustomException("Training dataframe is empty")

        if test_df.empty:
            raise CustomException("Test dataframe is empty")
        
        if "diagnosed_diabetes" not in train_df.columns:
            raise CustomException("Target column 'diagnosed_diabetes' missing in train dataset")
        
        if train_df.duplicated().sum() > 0:
            train_df.drop_duplicates(inplace=True)

        return train_df, test_df

    except Exception as e:
        raise CustomException(e, sys)
    
def save_processed(train_df, test_df):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    logger.info(f"Processed train saved -> {TRAIN_FILE}")
    logger.info(f"Processed test saved -> {TEST_FILE}")

def run_ingestion():
    logger.info("Loading raw data...")
    train_df, test_df = load_raw_data()

    logger.info("Running quality checks...")
    train_df, test_df = basic_quality_check(train_df, test_df)

    logger.info("Saving processed data...")
    save_processed(train_df, test_df)

    logger.info("Ingestion completed successfully.")

if __name__ == "__main__":
    run_ingestion()
    
