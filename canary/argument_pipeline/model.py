import datetime
import os
from pathlib import Path
from typing import Union

from joblib import dump, load
from sklearn.metrics import classification_report

from canary import __version__, logger
from canary.utils import MODEL_STORAGE_LOCATION


class Model:

    def __init__(self, model_id=None, model_storage_location=None, load=True):
        """
        :param model_id: the model of the ID. Cannot be none
        :param model_storage_location: the location of the models on the filesystem
        """
        self.model_id = model_id
        self.model = None
        self.trained_on = None
        self.metrics = {}

        # Model ID used as part of filename so
        # Should not be empty
        if model_id is None:
            raise ValueError("Model ID cannot be none")

        # Allow override
        if model_storage_location is None:
            model_storage_location = MODEL_STORAGE_LOCATION

        # Allow override
        if model_storage_location is not None:
            self.model_dir = model_storage_location

        # Try and make relevant directories
        os.makedirs(self.model_dir, exist_ok=True)

        # Try and load any present model that matches ID
        if load:
            self.load()
            if self.model is None:
                logger.warn("No model was loaded. Either download the pretrained ones or run the train() function")
        else:
            logger.info("Model loading was overridden. No model loaded.")

    def load(self, load_from: Path = None):
        """
        Load the model from disk. Currently doesn't allow loading a custom module.
        """

        file = Path(self.model_dir) / f"{self.model_id}.joblib"
        if os.path.isfile(file):
            model = load(file)
            if model and 'canary_version' in model:
                version = model['canary_version']
                if __version__ != version:
                    logger.warn(
                        "This model was trained using an older version of Canary. You may need to retrain the models!")
                try:
                    self.model_id = model['model_id']
                    self.model = model['model']
                    self.trained_on = model['trained_on']
                    self.metrics = model['metrics']
                except KeyError as key_error:
                    logger.error(f"There was an error loading the model: {key_error}.")

    def save(self, model_data: dict, save_to: Path = None):
        """
        Save the model to disk after training

        :param save_to:
        :param model_data:
        """

        model_data["trained_on"] = datetime.datetime.now()
        model_data["canary_version"] = __version__

        if save_to is None:
            dump(model_data, Path(self.model_dir) / f"{self.model_id}.joblib")
        else:
            dump(model_data, Path(save_to) / ".joblib")

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=False):
        logger.debug(f"Training of {self.__class__.__name__} has begun")
        pipeline_model.fit(train_data, train_targets)
        prediction = pipeline_model.predict(test_data)
        logger.debug(f"\nModel stats:\n{classification_report(prediction, test_targets)}")
        self.model = pipeline_model
        self.metrics = classification_report(prediction, test_targets, output_dict=True)

        if save_on_finish is True:
            model_data = {
                "model_id": self.model_id,
                "model": self.model,
                "metrics": self.metrics
            }
            self.save(model_data)

    def predict(self, data: Union[list, str], probability=False) -> Union[list, bool]:
        """
        Make a prediction

        :param data:
        :param probability:
        :return: a boolean or list of indicationg the prediction
        """
        if self.model is None:
            raise ValueError(
                "Cannot make a prediction because there are no models trained."
                " Either train a model or download the pretrained models.")

        data_type = type(data)

        def probability_predict(inp: Union[str, list]) -> Union[list[dict], dict]:
            """
            internal helper function to provide a nicer way of returning probability predictions
            """

            if type(inp) is list:
                predictions_list = []
                for i, item in enumerate(inp):
                    predictions_dict = {}
                    p = self.model.predict_proba([item])[0]
                    for j, class_ in enumerate(self.model.classes_):
                        predictions_dict[self.model.classes_[j]] = p[j]
                    predictions_list.append(predictions_dict)
                return predictions_list

            elif type(inp) is str:
                predictions_dict = {}
                p = self.model.predict_proba([inp])[0]
                for i, class_ in enumerate(self.model.classes_):
                    predictions_dict[self.model.classes_[i]] = p[i]
                return predictions_dict
            else:
                raise TypeError("Incorrect type passed to function")

        if data_type is str or data_type is list:
            if data_type is str:
                if probability is False:
                    prediction = self.model.predict([data])[0]
                    return prediction
                elif probability is True:
                    return probability_predict(data)
            elif data_type is list:
                predictions = []
                if probability is False:
                    for i, _ in enumerate(data):
                        predictions.append(self.model.predict([data[i]])[0])
                    return predictions
                else:
                    return probability_predict(data)
        else:
            raise TypeError("Expected a string or list as input")
