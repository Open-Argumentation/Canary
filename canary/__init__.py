import logging as _logging
import os as _os
from configparser import ConfigParser as _ConfigParser

import pkg_resources

# This logger should be used wherever possible
logger = _logging.getLogger(__name__)

# Set up the configuration parser.
# Should be used wherever possible
config = _ConfigParser()
config.read(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'etc/canary.cfg'))

# Handy to have. Similar to how pandas does it.
__version__ = pkg_resources.require("canary-am")[0].version


def load(*args, **kwargs):
    if len(args) == 1 and type(args[0]) is str:
        import canary.argument_pipeline
        return canary.argument_pipeline.load(*args, **kwargs)
    elif len(args) > 0:
        if all(type(s) is str for s in args):
            pass
        else:
            logger.error("load only accepts string parameters")
    else:
        logger.error("")
