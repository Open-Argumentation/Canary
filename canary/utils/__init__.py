"""
Utilities

that does things...
"""

import logging as _logging
import os as _os
from configparser import ConfigParser as _ConfigParser
from pathlib import Path as _Path

from .. import __file__ as _file

__all__ = [
    "logger",
    "config",
    "CANARY_ROOT_DIR",
    "CONFIG_LOCATION",
    "CANARY_LOCAL_STORAGE",
    "CANARY_MODEL_STORAGE_LOCATION",
    "CANARY_MODEL_DOWNLOAD_LOCATION",
    "CANARY_CORPORA_LOCATION"
]

# This logger should be used wherever possible
_logging.basicConfig(format="%(asctime)s:CANARY:%(levelname)s:%(message)s", datefmt='%d/%m/%Y-%H:%M:%S')
logger = _logging.getLogger(__name__)
"""

"""
# Set up the configuration parser.
# Should be used wherever possible
config = _ConfigParser()
"""
"""

config.read(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../etc/canary.cfg'))

# Various utility variables
CANARY_ROOT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(_file)))
"""The root directory of the Canary library"""

CONFIG_LOCATION = _Path(f"{CANARY_ROOT_DIR}/etc") / "canary.cfg"
"""The location of the default Canary configuration file"""

CANARY_LOCAL_STORAGE = _Path(f"{_Path.home()}/{config.get('canary', 'storage_location')}")
"""The location of the default canary storage location. Defaults to ~/.canary on linux and macOS"""

CANARY_MODEL_STORAGE_LOCATION = _Path(f"{CANARY_LOCAL_STORAGE}/models")
"""The default location where Canary models are saved to after training and downloading
"""
CANARY_MODEL_DOWNLOAD_LOCATION = config.get('canary', 'model_download_location')
CANARY_CORPORA_LOCATION = _os.path.join(_Path.home(), config.get('canary',
                                                                 'corpora_home_storage_directory'))
