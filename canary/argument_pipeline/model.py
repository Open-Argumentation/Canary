import os
import sys
from datetime import datetime
from logging import FileHandler
from pathlib import Path
from typing import Union

import joblib
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import canary
from canary.utils import CANARY_MODEL_STORAGE_LOCATION, CANARY_LOCAL_STORAGE


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

    def save(self, save_to: Path = None):
        """
        Save the model to disk after training

        :param save_to:
        """

        self._metadata = {
            "canary_version_trained_with": canary.__version__,
            "python_version_trained_with": tuple(sys.version_info),
            "trained_on": datetime.now()
        }

        if save_to is None:
            canary.utils.logger.info(f"Saving {self.model_id}")
            joblib.dump(self, Path(self.__model_dir) / f"{self.model_id}.joblib", compress=2)
        else:
            canary.utils.logger.info(f"Saving {self.model_id} to {save_to}.")
            joblib.dump(self, Path(save_to) / f"{self.model_id}.joblib", compress=2)

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

        # We need all of the below items to continue
        if any(item is None for item in [train_data, test_data, train_targets, test_targets]):

            # Check if we have a default training method
            if hasattr(self, "default_train"):
                canary.utils.logger.debug("Using default training method")
                train_data, test_data, train_targets, test_targets = self.default_train()

                return self.train(pipeline_model=pipeline_model, train_data=train_data, test_data=test_data,
                                  train_targets=train_targets, test_targets=test_targets, save_on_finish=save_on_finish,
                                  *args,
                                  **kwargs)
            else:
                raise ValueError(
                    "Missing required training / test data in method call. "
                    "There is no default training method for this model."
                    " Please supply these and try again.")

        canary.utils.logger.debug(f"Training of {self.__class__.__name__} has begun")

        if pipeline_model is None:
            pipeline_model = LogisticRegression(random_state=0)
            canary.utils.logger.warn("No model selected. Defaulting to Logistic Regression.")

        pipeline_model.fit(train_data, train_targets)
        prediction = pipeline_model.predict(test_data)

        report = f"\nModel stats:\n{classification_report(prediction, test_targets)}"

        if canary.utils.config.get('canary', 'dev') == "True":
            self._log_training_data(report)
        else:
            canary.utils.logger.debug(report)

        self._model = pipeline_model
        self._metrics = classification_report(prediction, test_targets, output_dict=True)

        if save_on_finish is True:
            self.save()

    def stratified_k_fold_train(self, pipeline_model=None, train_data=None, train_targets=None,
                                save_on_finish=True, n_splits=2, *args, **kwargs):

        if any(item is None for item in [train_data, train_data]):
            raise ValueError("...")

        if pipeline_model is None:
            pipeline_model = LogisticRegression(random_state=0)

        skf = StratifiedKFold(n_splits=n_splits)
        train_data = numpy.array(train_data)
        train_targets = numpy.array(train_targets)

        for train_index, test_index in skf.split(train_data, train_targets):
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_targets[train_index], train_targets[test_index]

            pipeline_model.fit(X_train.tolist(), y_train.tolist())
            prediction = pipeline_model.predict(X_test.tolist())
            report = f"\nModel stats:\n{classification_report(prediction, y_test.tolist())}"

            if canary.utils.config.get('canary', 'dev') == "True":
                self._log_training_data(report)
            else:
                canary.utils.logger.debug(report)
            self._metrics = classification_report(prediction, y_test.tolist(), output_dict=True)

        self._model = pipeline_model

        if save_on_finish is True:
            self.save()

    def _log_training_data(self, msg):
        training_log_dir = Path(CANARY_LOCAL_STORAGE) / "logs"
        os.makedirs(training_log_dir, exist_ok=True)

        training_log_dir = training_log_dir / "training.log"
        handler = FileHandler(filename=training_log_dir, encoding="utf-8")
        canary.utils.logger.addHandler(handler, )
        canary.utils.logger.debug(msg)
        canary.utils.logger.removeHandler(handler)

    def predict(self, data, probability=False) -> Union[list, bool]:
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
            canary.utils.logger.warn(
                f"This model doesn't support probability. Probability has been set to {probability}.")

        data_type = type(data)

        def probability_predict(inp) -> Union[list[dict], dict]:
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

            else:
                predictions_dict = {}
                p = self._model.predict_proba(inp)[0]
                for i, class_ in enumerate(self._model.classes_):
                    predictions_dict[self._model.classes_[i]] = p[i]
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
                return probability_predict(data)
