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
        from canary.utils import MODEL_STORAGE_LOCATION
        from pathlib import Path
        import os

        download_pretrained_models()
        self.assertTrue(os.path.isfile(Path(MODEL_STORAGE_LOCATION) / "models.zip"))

    def test_instantiate_argument_detector(self) -> None:
        """
        Test that we can instantiate the ArgumentDetector class without error.
        The model property should not be null.
        """
        from canary.argument_pipeline.classification import ArgumentDetector

        ad = ArgumentDetector()
        self.assertTrue(ad.model is not None)

    def test_instantiate_argument_component(self) -> None:
        """
        Test that we can instantiate the ArgumentComponent class without error.
        The model property should not be null.
        """
        from canary.argument_pipeline.component_identification import ArgumentComponent
        ag = ArgumentComponent()
        self.assertTrue(ag.model is not None)
