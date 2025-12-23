import argparse
from .utils import info
from .ingest import run_ingestion
from .train import run_training
from .ensemble import run_ensemble


def run_pipeline(args):
    if args.ingest:
        info("Running ingestion stage...")
        run_ingestion()

    if args.train:
        info("Running model training...")
        run_training()

    if args.ensemble:
        info("Running ensemble + submission generation...")
        run_ensemble()


def get_args():
    parser = argparse.ArgumentParser(description="Diabetes ML Pipeline")

    parser.add_argument("--ingest", action="store_true", help="Run ingestion pipeline")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--ensemble", action="store_true", help="Generate submission")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
