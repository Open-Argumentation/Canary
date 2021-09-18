import glob
from unittest import TestCase

import canary


class ArgumentPipeline(TestCase):
    """
    Tests for canary.argument_pipeline
    """

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.argument_detector = None
        self.structure_predictor = None
        self.argument_component = None
        self.arg_segmenter = None

    def setUp(self) -> None:
        """
        Setup for the tests.
        """
        from canary.argument_pipeline import download_pretrained_models

        download_pretrained_models("all")
        self.argument_detector = canary.load("argument_detector")
        self.structure_predictor = canary.load("structure_predictor")
        self.argument_component = canary.load("argument_component")
        self.arg_segmenter = canary.load("arg_segmenter")
        super().setUp()

    def test_base_classifier_instantiation_fails(self):
        """
        The base model should not be able to be instantiated. It's a base class.
        """

        from canary.argument_pipeline.model import Model
        with self.assertRaises(TypeError):
            _ = Model(model_id='model')

    def test_download_models(self) -> None:
        """
        Assert that the models has indeed downloaded.
        """
        from canary.utils import CANARY_MODEL_STORAGE_LOCATION

        models = glob.glob(str(CANARY_MODEL_STORAGE_LOCATION / "*.joblib"))
        self.assertTrue(len(models) > 0)

    def test_arg_detection_model_type(self):
        from canary.argument_pipeline.classification import ArgumentDetector

        self.assertTrue(type(self.argument_detector) is ArgumentDetector)

    def test_arg_segmenter_model_type(self):
        from canary.argument_pipeline.argument_segmenter import ArgumentSegmenter

        self.assertTrue(type(self.arg_segmenter) is ArgumentSegmenter)

    def test_arg_component_model_type(self):
        from canary.argument_pipeline.component_identification import ArgumentComponent

        self.assertTrue(type(self.argument_component) is ArgumentComponent)

    def test_arg_structure_model_type(self):
        from canary.argument_pipeline.structure_prediction import StructurePredictor

        self.assertTrue(type(self.structure_predictor) is StructurePredictor)

    def test_check_models_have_model_data(self):
        self.assertTrue(model._model is not None for model in
                        [self.arg_segmenter, self.argument_component, self.structure_predictor, self.argument_detector])
