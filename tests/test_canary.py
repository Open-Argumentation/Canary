from unittest import TestCase


class CanaryTests(TestCase):

    def test_version_not_none(self):
        from canary._version import __version__
        self.assertTrue(__version__ is not None and type(__version__) is str)
