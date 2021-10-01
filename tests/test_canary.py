from unittest import TestCase


class CanaryTests(TestCase):

    def __init__(self, methodName: str = ...) -> None:
        from canary.utils import config

        # convert to a pure dictionary
        self._config_items = {i: dict(config.items(i)) for i in config.sections()}
        super().__init__(methodName)

    def test_version_not_none(self):
        from canary._version import __version__
        self.assertTrue(__version__ is not None and type(__version__) is str)

    def test_config_top_level_is_present(self):
        """Check the very top level of the configuration is present"""
        from canary.utils import config

        # convert to a pure dictionary
        config_items = {i: dict(config.items(i)) for i in config.sections()}

        self.assertTrue("canary" in config_items and "nltk" in config_items)

    def test_canary_config_is_present(self):
        self.assertTrue(
            "corpora_home_storage_directory" in self._config_items['canary'] and
            "storage_location" in self._config_items["canary"] and
            "model_download_location" in self._config_items["canary"]
        )
