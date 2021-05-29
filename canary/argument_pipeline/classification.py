from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from canary.argument_pipeline.model import Model
from canary.corpora import load_ukp_sentential_argument_detection_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import CountPosVectorizer, DiscourseMatcher, CountPunctuationVectorizer, \
    LengthOfSentenceTransformer, SentimentTransformer, AverageWordLengthTransformer


class ArgumentDetector(Model):
    """
    Argument Detector

    Performs binary classification on text to determine if it is argumentative or not.
    """

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            self.model_id = "argument_detector"
        super().__init__(model_id=self.model_id,
                         model_storage_location=model_storage_location,
                         )

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        train_data, train_targets, test_data, test_targets = [], [], [], []
        for dataset in load_ukp_sentential_argument_detection_corpus(multiclass=False).values():
            for x in dataset['train']:
                test, target = x[0], x[1]
                test_data.append(test)
                test_targets.append(target)
            for y in dataset['test']:
                train, target = y[0], y[1]
                train_data.append(train)
                train_targets.append(target)

        model = Pipeline([
            ('features', FeatureUnion([
                ('pos_tagger', CountPosVectorizer(tokenizer=Lemmatizer())),
                ('bow',
                 CountVectorizer(
                     ngram_range=(1, 2),
                     tokenizer=Lemmatizer()
                 )
                 ),
                ("length", LengthOfSentenceTransformer()),
                ("discourse", DiscourseMatcher()),
                ("punctuation", CountPunctuationVectorizer()),
                ("sentiment", SentimentTransformer()),
                ("average_word_length", AverageWordLengthTransformer()),
            ])),
            ('SGDClassifier',
             SGDClassifier(
                 alpha=0.005,
                 n_jobs=2,
                 warm_start=True,
                 loss="log",
                 early_stopping=True,
                 random_state=0,
                 n_iter_no_change=7,
             )
             )
        ])
        super(ArgumentDetector, self).train(pipeline_model=model,
                                            train_data=train_data,
                                            test_data=test_data,
                                            train_targets=train_targets,
                                            test_targets=test_targets)
