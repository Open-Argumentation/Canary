import pkg_resources as __pkg_resources

# Handy to have. Similar to how pandas does it.
__version__ = __pkg_resources.require("canary-am")[0].version
