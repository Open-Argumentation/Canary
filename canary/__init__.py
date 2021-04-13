import logging
import os as _os
from configparser import ConfigParser as _ConfigParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
config = _ConfigParser()
config.read(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'etc/canary.cfg'))
