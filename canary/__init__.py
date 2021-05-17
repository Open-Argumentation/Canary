import logging
import os as _os
from configparser import ConfigParser as _ConfigParser
import importlib.metadata

# Set the logging level from the config file.
# This logger should be used wherever possible
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set up the configuration parser.
# Should be used wherever possible
config = _ConfigParser()
config.read(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'etc/canary.cfg'))

# Handy to have. Similar to how pandas does it.
__version__ = importlib.metadata.version('canary')