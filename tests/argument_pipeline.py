import glob
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
        download_pretrained_models("all")

    def test_download_models(self) -> None:
        """
        Assert that the models.zip file has indeed downloaded.
        """
        from canary.utils import CANARY_MODEL_STORAGE_LOCATION

        models = glob.glob(str(CANARY_MODEL_STORAGE_LOCATION / "*.joblib"))
        # test model dir created
        self.assertTrue(len(models) > 0)
