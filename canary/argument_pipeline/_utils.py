import json
import os
from pathlib import Path

import requests


def log_training_data(data: dict, file_name: str = None):
    """Internal helper method for logging training data.

    An internal function not designed to be used besides from logging training data.

    Parameters
    ----------
    data: dict
    file_name: str, optional

    """
    from canary.utils import CANARY_LOCAL_STORAGE

    training_log_dir = Path(CANARY_LOCAL_STORAGE) / "logs"
    os.makedirs(training_log_dir, exist_ok=True)

    if file_name is None:
        file_name = "training"

    training_log_file = training_log_dir / f"{file_name}.ndjson"

    with open(training_log_file, 'a') as file:
        file.write(json.dumps(data) + "\n")


def get_downloadable_assets_from_github():
    """Finds downloadable argument models which are available from Canary's GitHub repository

    Returns
    -------
    str
        The JSON string from the GitHub REST API
    """
    import canary.utils
    canary_repo = canary.utils.config.get('canary', 'model_download_location')
    res = requests.get(f"https://api.github.com/repos/{canary_repo}/releases/tags/latest",
                       headers={"Accept": "application/vnd.github.v3+json"})
    if res.status_code == 200:
        import json
        return json.loads(res.text)
    else:
        canary.utils.logger.error("Could not get details from github")
        return None


def get_models_not_on_disk():
    current_models = models_available_on_disk()
    github_assets = [j['name'].split(".")[0] for j in get_downloadable_assets_from_github()['assets']]
    return list(set(github_assets) - set(current_models))


def models_available_on_disk() -> list:
    from canary.utils import CANARY_MODEL_STORAGE_LOCATION

    from glob import glob
    return [Path(s).stem for s in glob(str(CANARY_MODEL_STORAGE_LOCATION / "*.joblib"))]
