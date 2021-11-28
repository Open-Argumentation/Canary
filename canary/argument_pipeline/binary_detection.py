from typing import Union

import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

from ..argument_pipeline.base import Model
from ..corpora import load_essay_corpus
from ..nlp.transformers import DiscourseMatcher, LengthOfSentenceTransformer, WordSentimentCounter, \
    AverageWordLengthTransformer, LengthTransformer, UniqueWordsTransformer

__all__ = [
    "ArgumentDetector"
]


class ArgumentDetector(Model):
    """Argument Detector

    Performs binary classification on text to determine if it is argumentative or not.

    Examples
    ---------
    >>> import canary
    >>> arg_detector = canary.load_model("argument_detector")
    >>> component = "The more body fat that you have, the greater your risk for heart disease"
    >>> print(arg_detector.predict(component))
    True

    >>> print(arg_detector.predict(component, probability=True))
    {False: 0.0, True: 1.0}
    """

    def __init__(self, model_id=None):
        if model_id is None:
            model_id = "argument_detector"

        super().__init__(model_id=model_id)

    @staticmethod
    def default_train():
        """Default training method which supplies the default training set"""
        from imblearn.over_sampling import RandomOverSampler

        ros = RandomOverSampler(random_state=0, sampling_strategy='not majority')
        x, y = load_essay_corpus(purpose="argument_detection")

        x, y = ros.fit_resample(pandas.DataFrame(x), pandas.DataFrame(y))

        return train_test_split(x.get(0).to_list(), y.get(0).to_list(),
                                train_size=0.5,
                                shuffle=True,
                                random_state=0,
                                stratify=y
                                )

    @classmethod
    def train(cls, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=False, *args, **kwargs):

        if "features" not in kwargs:
            feats = [
                CountVectorizer(
                    ngram_range=(1, 2),
                    lowercase=False,
                    max_features=2000,
                ),
                LengthOfSentenceTransformer(),
                AverageWordLengthTransformer(),
                UniqueWordsTransformer(),
                LengthTransformer(word_length=10),
                DiscourseMatcher('claim'),
                DiscourseMatcher('major_claim'),
                DiscourseMatcher('premise'),
                DiscourseMatcher('forward'),
                DiscourseMatcher('thesis'),
                DiscourseMatcher('rebuttal'),
                DiscourseMatcher('backward'),
                WordSentimentCounter(target="pos"),
                WordSentimentCounter(target="neg"),
            ]
        else:
            feats = kwargs.get('feats')

        if pipeline_model is None:
            pipeline_model = make_pipeline(
                make_union(
                    *feats
                ),
                MaxAbsScaler(),
                SVC(
                    random_state=0,
                    probability=True
                )
            )

            return super().train(pipeline_model=pipeline_model,
                                 train_data=train_data,
                                 test_data=test_data,
                                 train_targets=train_targets,
                                 test_targets=test_targets,
                                 save_on_finish=True)

        def predict(self, data, probability=False) -> Union[list, bool]:
            if type(data) is list:
                if not all(type(i) is str for i in data):
                    raise TypeError(f"{self.__class__.__name__} requires list elements to be strings.")

            return super().predict(data, probability)
