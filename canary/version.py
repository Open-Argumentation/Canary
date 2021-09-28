import pkg_resources as __pkg_resources

__all__ = ["__version__"]

# Handy to have. Similar to how pandas does it.
try:
    __version__ = __pkg_resources.require("canary-am")[0].version
except:
    __version__ = None
