import os
from pathlib import Path
from configparser import ConfigParser

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CONFIG_LOCATION = f"{ROOT_DIR}/etc/canary.cfg"

__config = ConfigParser()
__config.read(CONFIG_LOCATION)

CANARY_LOCAL_STORAGE = os.path.join(Path.home(), __config.get("canary", "storage_location"))
