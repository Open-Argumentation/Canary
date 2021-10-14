import os
import sys
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Union

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .. import __version__
from ..utils import CANARY_MODEL_STORAGE_LOCATION, logger, get_is_dev

__all__ = [
    "Model"
]


class Model(metaclass=ABCMeta):
    """Abstract class that other Canary models descend from
    """

    @abstractmethod
    def __init__(self, model_id=None):
        """Constructor method

        Parameters
        ----------
        model_id: str
            The model ID. The model ID must never be None.
        """

        self.__model_id = model_id
        self._model = None
        self._metrics = {}
        self._metadata = {}
        self.__model_dir = CANARY_MODEL_STORAGE_LOCATION

        # Model ID used as part of filename so
        # Should not be empty
        if model_id is None:
            raise ValueError("Model ID cannot be none")

        # Try and make relevant directories
        os.makedirs(self.__model_dir, exist_ok=True)

    def __repr__(self):
        return self.model_id

    @property
    def model_id(self):
        """Returns the model id

        Returns
        -------
        str
            The model id
        """

        return self.__model_id

    @property
    def supports_probability(self):
        """
        Returns a boolean if the model supports probability prediction.

        Returns
        -------
        str
            The boolean value indicating if probability predictions are possible.
        """

        if hasattr(self._model, 'predict_proba'):
            return True
        return False

    def fit(self, training_data: list, training_labels: list):
        """Fits a model to the training data.

        Parameters
        ----------
        training_data: list
            The training data on which the model is fitted
        training_labels: list:
            The training labels on which the data is fitted to

        Returns
        -------
        self
        """
        self._model.fit(training_data, training_labels)
        return self

    @property
    def metrics(self):
        """Property which returns model metrics

        Returns
        -------
        dict
            Returns the metrics of the model as a dict

        Examples
        --------
        >>> self.metrics
        {"f1score" 54.6, ...}
        """
        return self._metrics

    def set_model(self, model):
        """Set the scikit-learn model that sits under self._model

        Parameters
        ----------
        model
            a model that conforms to the standard scikit-learn API
        """
        self._model = model

    def save(self, save_to: Path = None):
        """Saves the model to disk after training

        Parameters
        ----------
        save_to: str
            Where to save the model
        """

        self._metadata = {
            "canary_version_trained_with": __version__,
            "python_version_trained_with": tuple(sys.version_info),
            "trained_on": datetime.now()
        }

        if save_to is None:
            logger.info(f"Saving {self.model_id}")
            joblib.dump(self, Path(self.__model_dir) / f"{self.model_id}.joblib", compress=2)
        else:
            logger.info(f"Saving {self.model_id} to {save_to}.")
            joblib.dump(self, Path(save_to) / f"{self.model_id}.joblib", compress=2)

    @classmethod
    def train(cls, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):
        """Classmethod which initialises a model and trains it on the provided training data.

        Parameters
        ----------
        pipeline_model
            The model which is trained to make predictions
        train_data: list
            Training data
        test_data: list
            Test data
        train_targets: list
            The training labels
        test_targets: list
            The test labels
        save_on_finish: bool
            Should the model be saved when training has finished?
        *args: tuple
            Additional positional arguments
        **kwargs: dict
            Additional keyed-arguments

        Returns
        -------
        Model
            The model instance
        """

        model = cls()

        # We need all of the below items to continue
        if any(item is None for item in [train_data, test_data, train_targets, test_targets]):

            # Check if we have a default training method
            if hasattr(model, "default_train"):
                logger.debug("Using default training method")
                train_data, test_data, train_targets, test_targets = model.default_train()

                return model.train(pipeline_model=pipeline_model, train_data=train_data, test_data=test_data,
                                   train_targets=train_targets, test_targets=test_targets,
                                   save_on_finish=save_on_finish,
                                   *args,
                                   **kwargs)
            else:
                raise ValueError(
                    "Missing required training / test data in method call. "
                    "There is no default training method for this model."
                    " Please supply these and try again.")

        logger.debug(f"Training of {model.__class__.__name__} has begun")

        if pipeline_model is None:
            pipeline_model = LogisticRegression(random_state=0)
            logger.warn("No model selected. Defaulting to Logistic Regression.")

        model.set_model(pipeline_model)
        model.fit(train_data, train_targets)

        prediction = model.predict(test_data)

        if get_is_dev() is True:
            from ._utils import log_training_data
            log_training_data({"result": classification_report(test_targets, prediction, output_dict=True),
                               "datetime": str(datetime.now()), "model": model.__class__.__name__})

        logger.debug(f"\nModel stats:\n{classification_report(test_targets, prediction)}")

        model._metrics = classification_report(test_targets, prediction, output_dict=True)

        if save_on_finish is True:
            model.save()

        return model

    def predict(self, data, probability=False) -> Union[list, bool]:
        """Make a prediction on some data. A wrapper around scikit-learn's predict method.

        Parameters
        ----------
        data:
            The data the predictor will be ran on.
        probability: bool
            boolean indicating if the method should return a probability prediction.

        Notes
        ------
        Not all models support probability predictions. This can be checked with the supports_probability property.

        Returns
        -------
        Union[list, bool]
            a boolean indicating the predictions or list of predictions
        """

        if self._model is None:
            raise ValueError(
                "Cannot make a prediction because no model has been loaded."
                " Either train a model or download the pretrained models.")

        if self.supports_probability is False and probability is True:
            probability = False
            logger.warn(
                f"This model doesn't support probability. Probability has been set to {probability}.")

        data_type = type(data)

        def probability_predict(inp) -> Union[list[dict], dict]:
            """Internal helper function to provide a nicer way of returning probability predictions.

            The default 'predict_proba' returns positional floats which requires that you know
            the ordering of the classes. This returns the labels along with the float value.
            """

            if type(inp) is list:
                predictions_list = []
                for _, item in enumerate(inp):
                    predictions_dict = {}
                    p = self._model.predict_proba([item])[0]
                    for j, class_ in enumerate(self._model.classes_):
                        predictions_dict[self._model.classes_[j]] = p[j]
                    predictions_list.append(predictions_dict)
                return predictions_list

            else:
                predictions_dict = {}
                p = self._model.predict_proba(inp)[0]
                for _i, class_ in enumerate(self._model.classes_):
                    predictions_dict[self._model.classes_[_i]] = p[_i]
                return predictions_dict

        if data_type is list:
            predictions = []
            if probability is False:
                for i, _ in enumerate(data):
                    predictions.append(self._model.predict([data[i]])[0])
                return predictions
            else:
                return probability_predict(data)
        else:
            if probability is False:
                prediction = self._model.predict([data])[0]
                return prediction
            elif probability is True:
                return probability_predict([data])[0]
