from typing import Union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from ..argument_pipeline.base import Model
from ..corpora import load_essay_corpus
from ..nlp import Lemmatiser
from ..nlp.transformers import DiscourseMatcher, CountPunctuationVectorizer, \
    LengthOfSentenceTransformer, SentimentTransformer, AverageWordLengthTransformer, WordSentimentCounter

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
        x, y = load_essay_corpus(purpose="argument_detection")
        return train_test_split(x, y,
                                train_size=0.7,
                                shuffle=True,
                                random_state=0,
                                stratify=y
                                )

    @classmethod
    def train(cls, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=False, *args, **kwargs):

        if pipeline_model is None:
            pipeline_model = Pipeline([
                ('features', FeatureUnion([
                    ('bow',
                     CountVectorizer(
                         ngram_range=(1, 2),
                         tokenizer=Lemmatiser(),
                         lowercase=False
                     )
                     ),
                    ("length", LengthOfSentenceTransformer()),
                    ("support", DiscourseMatcher(component="support")),
                    ("conflict", DiscourseMatcher(component="conflict")),
                    ("punctuation", CountPunctuationVectorizer()),
                    ("sentiment", SentimentTransformer()),
                    ("sentiment_pos", WordSentimentCounter(target="pos")),
                    ("sentiment_neg", WordSentimentCounter(target="neg")),
                    ("average_word_length", AverageWordLengthTransformer()),
                ])),
                ('SGDClassifier',
                 SGDClassifier(
                     class_weight='balanced',
                     random_state=0,
                     loss='modified_huber',
                 )
                 )
            ])

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
