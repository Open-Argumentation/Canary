import datetime
import os
from pathlib import Path
from joblib import dump, load
from sklearn.metrics import classification_report
from canary.utils import MODEL_STORAGE_LOCATION
from canary import logger
from typing import Union
from canary import __version__


class Model:
    
    def __init__(self, model_id=None, model_storage_location=None):
        """
        :param model_id: the model of the ID. Cannot be none
        :param model_storage_location: the location of the models on the filesystem
        """

        self.model_id = model_id
        self.model = None
        self.trained_on = None

        if model_id is None:
            raise ValueError("Model ID cannot be none")

        if model_storage_location is None:
            model_storage_location = MODEL_STORAGE_LOCATION

        # Allow override
        if model_storage_location is not None:
            self.model_dir = model_storage_location

        os.makedirs(self.model_dir, exist_ok=True)

        self.load()
        if self.model is None:
            logger.warn("No model was loaded. Either download the pretrained ones or run the train() function")

    def load(self):
        """
        Load the model from disk. Currently doesn't allow loading a custom module.
        """

        file = Path(self.model_dir) / f"{self.model_id}.joblib"
        if os.path.isfile(file):
            model = load(file)
            if model and 'canary_version' in model:
                version = model['canary_version']
                if __version__ == version:
                    self.model_id = model['model_id']
                    self.model = model['model']
                    self.trained_on = model['trained_on']
                else:
                    logger.warn(
                        "This model was trained using an older version of Canary. You may need to retrain the models!")

    def save(self, model_data: dict):
        """
        Save the model to disk after training

        :param model_data:
        """

        dump(model_data, Path(self.model_dir) / f"{self.model_id}.joblib")

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        logger.debug(f"Training of {self.__class__.__name__} has begun")
        pipeline_model.fit(train_data, train_targets)
        prediction = pipeline_model.predict(test_data)
        logger.debug(f"\nModel stats:\n{classification_report(prediction, test_targets)}")
        self.model = pipeline_model

        model_data = {
            "model_id": self.model_id,
            "model": self.model,
            "trained_on": datetime.datetime.now(),
            "canary_version": __version__
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
            raise ValueError("No model trained. Cannot make predictions")

        data_type = type(data)

        if data_type is str or data_type is list:
            if data_type is str:
                if probability is False:
                    prediction = self.model.predict([data])[0]
                    return prediction
                elif probability is True:
                    prediction = self.model.predict_proba([data])[0]
                    return prediction
            elif data_type is list:
                predictions = []
                for i, _ in enumerate(data):
                    if probability is False:
                        predictions.append(self.model.predict([data[i]])[0])
                    else:
                        predictions.append(self.model.predict_proba([data[i]])[0])
                return predictions
        else:
            raise TypeError("Expected a string or list as input")
