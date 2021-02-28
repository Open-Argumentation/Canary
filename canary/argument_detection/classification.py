import os

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

from canary.corpora import load_ukp_sentential_argument_detection_corpus
from canary.utils import CANARY_LOCAL_STORAGE
from canary import logger
from datetime import datetime
from canary.data.indicators import discourse_indicators
from joblib import dump, load

from canary.preprocessing import Preprocessor


class LengthTransformer(object):

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[len(y) > 12] for y in x]


class DiscourseMatcher(object):

    @property
    def indicators(self):
        indicators = discourse_indicators['claim'] + discourse_indicators['major_claim'] + discourse_indicators[
            'premise']
        return indicators

    def fit(self, x, y):
        return self

    def transform(self, doc):
        return [[x in self.indicators] for x in doc]


class ArgumentDetector:
    __model_dir = f"{CANARY_LOCAL_STORAGE}/models"

    def __init__(self, method="naive_bayes", pre_train=True, model_storage_location=None, force_retrain=False,
                 domain="auto"):

        self.method = method
        self.pre_train = pre_train
        self.__model = None
        self.model_id = f"arg_detection_{self.method}"

        # Allow override
        if model_storage_location is not None:
            self.__model_dir = model_storage_location

        os.makedirs(self.__model_dir, exist_ok=True)

        if self.pre_train is True:
            self.__load_classifier__()
            if self.__model is None or force_retrain is True:
                self.__model = self.__train_model__()

    def __save_classifier__(self, model_data):
        dump(model_data, f"{self.__model_dir}/{self.model_id}.joblib")

    def __load_classifier__(self):
        file = f"{self.__model_dir}/{self.model_id}.joblib"
        if os.path.isfile(file):
            model = load(f"{self.__model_dir}/{self.model_id}.joblib")
            if model:
                self.__model = model['model']

    def __train_model__(self):
        train_data, train_targets, test_data, test_targets = [], [], [], []
        for dataset in load_ukp_sentential_argument_detection_corpus().values():

            for x in dataset['train']:
                test, target = x[0], x[1]
                if target == 'NoArgument':
                    target = False
                else:
                    target = True
                test_data.append(test)
                test_targets.append(target)
            for y in dataset['test']:
                train, target = y[0], y[1]
                if target == 'NoArgument':
                    target = False
                else:
                    target = True
                train_data.append(train)
                train_targets.append(target)

        model = Pipeline([
            ('feats', FeatureUnion([
                ('countVectorizer', CountVectorizer(ngram_range=(1, 3), stop_words=Preprocessor().stopwords)),
                ('length', LengthTransformer()),
                ('wp', CountVectorizer(ngram_range=(2, 2))),
                ("di", DiscourseMatcher())
            ])),
            ('clf', ComplementNB())
        ])
        model.fit(train_data, train_targets)
        prediction = model.predict(test_data)
        logger.debug(f"\nModel stats:\n{classification_report(prediction, test_targets)}")

        model_data = {
            "model_id": self.model_id,
            "model": model,
            "algorithm": self.method,
            "trained": datetime.now(),
        }

        self.__save_classifier__(model_data)
        return model

    def detect(self, corpora: list) -> list:
        predictions = []

        for doc in corpora:
            predictions.append((doc, self.predict(doc)))
        return predictions

    def predict(self, sentence: str) -> bool:
        return self.__model.predict([sentence])
