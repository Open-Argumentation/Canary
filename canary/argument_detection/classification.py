import os

from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB
from canary.download import load_imdb_debater_evidence_sentences
from canary.utils import CANARY_LOCAL_STORAGE
from datetime import datetime
from joblib import dump, load
from canary import logger

from canary.preprocessing import Vectorizer


class ArgumentDetector:
    __model_dir = f"{CANARY_LOCAL_STORAGE}/models"

    def __init__(self, method="naive_bayes", pre_train=True, model_storage_location=None, force_retrain=False):
        self.method = method
        self.pre_train = pre_train
        self.__model = None
        self.model_id = f"arg_detection_{self.method}"

        # Allow override
        if model_storage_location is not None:
            self.__model_dir = model_storage_location

        os.makedirs(self.__model_dir, exist_ok=True)

        if self.pre_train is True:
            self.__model = self.__load_classifier__()
            if self.__model is None or force_retrain is True:
                self.__model = self.__train_model__()

    def __save_classifier__(self, model_data):
        dump(model_data, f"{self.__model_dir}/{self.model_id}.joblib")

    def __load_classifier__(self):
        file = f"{self.__model_dir}/{self.model_id}.joblib"
        if os.path.isfile(file):
            model = load(f"{self.__model_dir}/{self.model_id}.joblib")
            return model['model']
        else:
            return None

    def __train_model__(self):
        train_data, train_targets, test_data, test_targets = load_imdb_debater_evidence_sentences()
        model = ComplementNB()
        v = Vectorizer().vectoriser()
        train_bow = v.fit_transform(train_data)
        test_bow = v.transform(test_data)
        model.fit(train_bow, train_targets)
        prediction = model.predict(test_bow)
        logger.debug("Model stats:")
        logger.debug(classification_report(prediction, test_targets))

        model_data = {
            "model_id": self.model_id,
            "model": model,
            "algorithm": self.method,
            "trained": datetime.now(),
        }

        self.__save_classifier__(model_data)
        return model

    def detect(self, corpora) -> bool:
        pass
