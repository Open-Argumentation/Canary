from unittest import TestCase


class ArgumentPipeline(TestCase):
    """
    Tests for canary.argument_pipeline
    """

    def setUp(self) -> None:
        """
        Setup for the tests.
        """
        from canary.argument_pipeline import download_pretrained_models

        super().setUp()
        download_pretrained_models()

    def test_download_models(self) -> None:
        """
        Assert that the models.zip file has indeed downloaded.
        """
        from canary.argument_pipeline import download_pretrained_models
        from canary.utils import CANARY_MODEL_STORAGE_LOCATION
        from pathlib import Path
        import os

        download_pretrained_models()
        self.assertTrue(os.path.isfile(Path(CANARY_MODEL_STORAGE_LOCATION) / "models.zip"))


