import json
import os
from pathlib import Path

from canary.utils import CANARY_LOCAL_STORAGE


def log_training_data(data: dict, file_name: str = None):
    """Internal helper method for logging training data.

    An internal function not designed to be used besides from logging training data.

    Parameters
    ----------
    data: dict
    file_name: str, optional

    """
    training_log_dir = Path(CANARY_LOCAL_STORAGE) / "logs"
    os.makedirs(training_log_dir, exist_ok=True)

    if file_name is None:
        file_name = "training"

    training_log_file = training_log_dir / f"{file_name}.ndjson"

    with open(training_log_file, 'a') as file:
        file.write(json.dumps(data) + "\n")
