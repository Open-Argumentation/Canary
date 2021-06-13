from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import DiscourseMatcher, \
    LengthTransformer, LengthOfSentenceTransformer, \
    AverageWordLengthTransformer, UniqueWordsTransformer, CountPosVectorizer, \
    WordSentimentCounter, TfidfPunctuationVectorizer


class ArgumentComponent(Model):
    """
    Detects argumentative components from natural language
    """

    def __init__(self, model_id: str = None, model_storage_location=None, load: bool = True):
        """
        :param model_id: the ID of the model
        :param model_storage_location: where the model should be stored
        :param load: Whether to automatically load the model
        """

        if model_id is None:
            self.model_id = "argument_component"
        else:
            self.model_id = model_id

        super().__init__(
            model_id=self.model_id,
            model_storage_location=model_storage_location,
            load=load
        )

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        train_data, test_data, train_targets, test_targets = load_essay_corpus()

        estimators = [
            ("RandomForestClassifier", RandomForestClassifier(
                random_state=0,
                n_jobs=2,
                warm_start=True,
            )),
            ('SGDClassifier', SGDClassifier(
                random_state=0,
                alpha=0.0005,
                early_stopping=True,
                tol=1e-5,
                n_jobs=2,
                n_iter_no_change=7,
                max_iter=999999999,
            )),
            ('SVC', SVC(
                gamma='auto',
                kernel='linear',
                random_state=0,
                C=9000000,
                max_iter=-1,
            )),
        ]

        model = Pipeline([
            ('feats', FeatureUnion([
                ('tfidvectorizer',
                 TfidfVectorizer(ngram_range=(1, 4), tokenizer=Lemmatizer(), lowercase=False)),
                ('length_5', LengthTransformer()),
                ('length_10', LengthTransformer(12)),
                ('CountPunctuationVectorizer', TfidfPunctuationVectorizer()),
                ('contain_claim', DiscourseMatcher('claim')),
                ('contain_premise', DiscourseMatcher('premise')),
                ('contain_major_claim', DiscourseMatcher('major_claim')),
                ('length_of_sentence', LengthOfSentenceTransformer()),
                ('average_word_length', AverageWordLengthTransformer()),
                ("eee", CountPosVectorizer()),
                ("eeeareae", UniqueWordsTransformer()),
                ("pos_sent", WordSentimentCounter("pos")),
                ("neg_sent", WordSentimentCounter("neg")),
            ])),
            ('clf', StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(
                    warm_start=True,
                    random_state=0,
                    max_iter=99999,
                )
            ))
        ])

        super(ArgumentComponent, self).train(pipeline_model=model,
                                             train_data=train_data,
                                             test_data=test_data,
                                             train_targets=train_targets,
                                             test_targets=test_targets)
