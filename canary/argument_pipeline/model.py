import datetime
import os
import zipfile
from pathlib import Path
from joblib import dump, load
from sklearn.metrics import classification_report

from canary.utils import MODEL_STORAGE_LOCATION
from canary import logger


class Model:
    model = None
    model_id = None

    def __init__(self, model_id=None, model_storage_location=None):
        self.model_id = model_id
        if model_storage_location is None:
            model_storage_location = MODEL_STORAGE_LOCATION

        # Allow override
        if model_storage_location is not None:
            self.model_dir = model_storage_location

        os.makedirs(self.model_dir, exist_ok=True)

        self.__load__()

    def __load__(self):
        file = Path(self.model_dir) / f"{self.model_id}.joblib"
        if os.path.isfile(file):
            model = load(file)
            if model:
                self.model = model['model']

    def __save__(self, model_data):
        dump(model_data, Path(self.model_dir) / f"{self.model_id}.joblib")

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        pipeline_model.fit(train_data, train_targets)
        prediction = pipeline_model.predict(test_data)
        logger.debug(f"\nModel stats:\n{classification_report(prediction, test_targets)}")

        model_data = {
            "model_id": self.model_id,
            "model": self.model,
            "trained_on": datetime.datetime.now()
        }
        self.__save__(model_data)

    def detect(self, corpora: list) -> list:
        predictions = []

        for doc in corpora:
            predictions.append((doc, self.predict(doc)))
        return predictions

    def predict(self, sentence: str, probability=False) -> bool:
        if probability is False:
            return self.model.predict([sentence])[0]
        else:
            return self.model.predict_proba([sentence])[0]
