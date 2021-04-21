import os
from pathlib import Path
from canary import config

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CONFIG_LOCATION = Path(f"{ROOT_DIR}/etc") / "canary.cfg"

CANARY_LOCAL_STORAGE = Path(f"{Path.home()}/{config.get('canary', 'storage_location')}")
MODEL_STORAGE_LOCATION = Path(f"{CANARY_LOCAL_STORAGE}/models")
MODEL_DOWNLOAD_LOCATION = config.get('canary', 'model_download_location')
CANARY_CORPORA_LOCATION = os.path.join(Path.home(), config.get('canary',
                                                              'corpora_home_storage_directory'))