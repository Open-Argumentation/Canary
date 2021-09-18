from typing import Union

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from canary.argument_pipeline.model import Model
from canary.corpora import load_ukp_sentential_argument_detection_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import DiscourseMatcher, CountPunctuationVectorizer, \
    LengthOfSentenceTransformer, SentimentTransformer, AverageWordLengthTransformer, WordSentimentCounter


class ArgumentDetector(Model):
    """
    Argument Detector

    Performs binary classification on text to determine if it is argumentative or not.
    """

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            model_id = "argument_detector"

        super().__init__(model_id=model_id,
                         model_storage_location=model_storage_location,
                         )

    @staticmethod
    def default_train():
        train_data = []
        test_data = []
        train_targets = []
        test_targets = []

        for dataset in load_ukp_sentential_argument_detection_corpus(multiclass=False).values():
            for x in dataset['train']:
                test, target = x[0], x[1]
                test_data.append(test)
                test_targets.append(target)
            for y in dataset['test']:
                train, target = y[0], y[1]
                train_data.append(train)
                train_targets.append(target)
        return train_data, test_data, train_targets, test_targets

    @classmethod
    def train(cls, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=False, *args, **kwargs):

        if pipeline_model is None:
            pipeline_model = Pipeline([
                ('features', FeatureUnion([
                    ('bow',
                     CountVectorizer(
                         ngram_range=(1, 2),
                         tokenizer=Lemmatizer(),
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
