import glob
from unittest import TestCase

import canary


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
        Assert that the models has indeed downloaded.
        """
        from canary.utils import CANARY_MODEL_STORAGE_LOCATION

        models = glob.glob(str(CANARY_MODEL_STORAGE_LOCATION / "*.joblib"))
        self.assertTrue(len(models) > 0)

    def test_arg_detection_model_type(self):
        argument_detector = canary.load("argument_detector")
        from canary.argument_pipeline.classification import ArgumentDetector

        self.assertTrue(type(argument_detector) is ArgumentDetector)

    def test_arg_segmenter_model_type(self):
        argument_detector = canary.load("arg_segmenter")
        from canary.argument_pipeline.argument_segmenter import ArgumentSegmenter

        self.assertTrue(type(argument_detector) is ArgumentSegmenter)

    def test_arg_component_model_type(self):
        argument_detector = canary.load("argument_component")
        from canary.argument_pipeline.component_identification import ArgumentComponent

        self.assertTrue(type(argument_detector) is ArgumentComponent)

    def test_arg_structure_model_type(self):
        argument_detector = canary.load("structure_predictor")
        from canary.argument_pipeline.structure_prediction import StructurePredictor

        self.assertTrue(type(argument_detector) is StructurePredictor)

