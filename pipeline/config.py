"""Shared configuration for the final experiment."""
from pathlib import Path
import os

RANDOM_STATE = 42
MIN_WORDS = 50
MIN_REQUIRED = 1000
CV_FOLDS = 5
MAX_FEATURES = 5000

# The final training file preserves this append order.
ORIGINAL_TRAINING_ROWS = 3402
HC3_END_ROW = 5402

LOCAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def configured_path(environment_name: str, filename: str) -> Path:
    """Use an environment override, a local file, or the original output folder."""
    override = os.environ.get(environment_name)
    if override:
        return Path(override)
    local_file = Path(filename)
    if local_file.exists():
        return local_file
    return LOCAL_DATA_DIR / filename


TRAIN_DATA_PATH = configured_path(
    "NLP_TRAIN_DATA_PATH",
    "AI vs Human Training Set.csv",
)
TEST_DATA_PATH = configured_path(
    "NLP_TEST_DATA_PATH",
    "AI vs Human Test Set.csv",
)
CHALLENGE_DATA_PATH = configured_path(
    "NLP_CHALLENGE_DATA_PATH",
    "AI vs Human Test Set (edge cases).csv",
)
OUTPUT_DIR = Path(os.environ.get("NLP_OUTPUT_DIR", "nlp_project_outputs"))

