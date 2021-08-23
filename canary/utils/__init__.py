import logging as _logging
import os as _os
from configparser import ConfigParser as _ConfigParser
from pathlib import Path as _Path

import canary

# This logger should be used wherever possible
logger = _logging.getLogger(__name__)

# Set up the configuration parser.
# Should be used wherever possible
config = _ConfigParser()
config.read(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../etc/canary.cfg'))

# Various utility variables
CANARY_ROOT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(canary.__file__)))
CONFIG_LOCATION = _Path(f"{CANARY_ROOT_DIR}/etc") / "canary.cfg"
CANARY_LOCAL_STORAGE = _Path(f"{_Path.home()}/{config.get('canary', 'storage_location')}")
CANARY_MODEL_STORAGE_LOCATION = _Path(f"{CANARY_LOCAL_STORAGE}/models")
CANARY_MODEL_DOWNLOAD_LOCATION = config.get('canary', 'model_download_location')
CANARY_CORPORA_LOCATION = _os.path.join(_Path.home(), config.get('canary',
                                                                 'corpora_home_storage_directory'))
