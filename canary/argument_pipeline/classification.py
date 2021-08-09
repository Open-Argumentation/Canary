from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from canary.argument_pipeline.model import Model
from canary.corpora import load_ukp_sentential_argument_detection_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import CountPosVectorizer, DiscourseMatcher, CountPunctuationVectorizer, \
    LengthOfSentenceTransformer, SentimentTransformer, AverageWordLengthTransformer, WordSentimentCounter


class ArgumentDetector(Model):
    """
    Argument Detector

    Performs binary classification on text to determine if it is argumentative or not.
    """

    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "argument_detector"

        super().__init__(model_id=self.model_id,
                         model_storage_location=model_storage_location,
                         load=load
                         )

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=False, *args, **kwargs):
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
                ('bow',
                 CountVectorizer(
                     ngram_range=(1, 3),
                     tokenizer=Lemmatizer(),
                     lowercase=False
                 )
                 ),
                ('pos_tagger', CountPosVectorizer()),
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

        super(ArgumentDetector, self).train(pipeline_model=model,
                                            train_data=train_data,
                                            test_data=test_data,
                                            train_targets=train_targets,
                                            test_targets=test_targets,
                                            save_on_finish=True)
