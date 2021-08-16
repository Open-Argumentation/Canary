import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import joblib
from sklearn.metrics import classification_report

from canary import __version__, logger
from canary.utils import CANARY_MODEL_STORAGE_LOCATION


class Model:

    def __init__(self, model_id=None, model_storage_location=None):
        """
        :param model_id: the model of the ID. Cannot be none
        :param model_storage_location: the location of the models on the filesystem
        """

        self.__model_id = model_id
        self._model = None
        self._metrics = {}
        self._metadata = {}

        # Model ID used as part of filename so
        # Should not be empty
        if model_id is None:
            raise ValueError("Model ID cannot be none")

        # Allow override
        if model_storage_location is None:
            model_storage_location = CANARY_MODEL_STORAGE_LOCATION

        # Allow override
        if model_storage_location is not None:
            self.__model_dir = model_storage_location

        # Try and make relevant directories
        os.makedirs(self.__model_dir, exist_ok=True)

    def __repr__(self):
        return self.model_id

    @property
    def model_id(self):
        return self.__model_id

    @property
    def supports_probability(self):
        if hasattr(self._model, 'predict_proba'):
            return True
        return False

    @property
    def metrics(self):
        return self._metrics

    def save(self, additional_data: dict = None, save_to: Path = None):
        """
        Save the model to disk after training

        :param save_to:
        :param additional_data:
        """

        self._metadata = {
            "canary_version_trained_with": __version__,
            "python_version_trained_with": tuple(sys.version_info),
            "trained_on": datetime.now()
        }

        if save_to is None:
            joblib.dump(self, Path(self.__model_dir) / f"{self.model_id}.joblib")
        else:
            joblib.dump(self, Path(save_to) / f"{self.model_id}.joblib")

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):
        """
        :param pipeline_model:
        :param train_data:
        :param test_data:
        :param train_targets:
        :param test_targets:
        :param save_on_finish:
        :param args:
        :param kwargs:
        :return:
        """

        logger.debug(f"Training of {self.__class__.__name__} has begun")

        pipeline_model.fit(train_data, train_targets)
        prediction = pipeline_model.predict(test_data)
        logger.debug(f"\nModel stats:\n{classification_report(prediction, test_targets)}")
        self._model = pipeline_model
        self._metrics = classification_report(prediction, test_targets, output_dict=True)

        if save_on_finish is True:
            self.save()

    def predict(self, data: Union[list, str], probability=False) -> Union[list, bool]:
        """
        Make a prediction

        :param data:
        :param probability:
        :return: a boolean or list of indi the prediction
        """

        if self._model is None:
            raise ValueError(
                "Cannot make a prediction because no model has been loaded."
                " Either train a model or download the pretrained models.")

        if self.supports_probability is False and probability is True:
            probability = False
            logger.warn(f"This model doesn't support probability. Probability has been set to {probability}.")

        data_type = type(data)

        def probability_predict(inp: Union[str, list]) -> Union[list[dict], dict]:
            """
            internal helper function to provide a nicer way of returning probability predictions
            """

            if type(inp) is list:
                predictions_list = []
                for i, item in enumerate(inp):
                    predictions_dict = {}
                    p = self._model.predict_proba([item])[0]
                    for j, class_ in enumerate(self._model.classes_):
                        predictions_dict[self._model.classes_[j]] = p[j]
                    predictions_list.append(predictions_dict)
                return predictions_list

            elif type(inp) is str:
                predictions_dict = {}
                p = self._model.predict_proba([inp])[0]
                for i, class_ in enumerate(self._model.classes_):
                    predictions_dict[self._model.classes_[i]] = p[i]
                return predictions_dict
            else:
                raise TypeError("Incorrect type passed to function")

        if data_type is str or data_type is list:
            if data_type is str:
                if probability is False:
                    prediction = self._model.predict([data])[0]
                    return prediction
                elif probability is True:
                    return probability_predict(data)
            elif data_type is list:
                predictions = []
                if probability is False:
                    for i, _ in enumerate(data):
                        predictions.append(self._model.predict([data[i]])[0])
                    return predictions
                else:
                    return probability_predict(data)
        else:
            raise TypeError("Expected a string or list as input")
