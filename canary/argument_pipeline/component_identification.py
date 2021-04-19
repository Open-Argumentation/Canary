from sklearn.linear_model import LogisticRegression

from canary.argument_pipeline.classification import ArgumentDetector
from canary.preprocessing import Preprocessor, Lemmatizer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from canary.corpora import load_essay_corpus
from canary.preprocessing.transformers import TfidfPosVectorizer, DiscourseMatcher, \
    FirstPersonIndicatorMatcher, TfidfPunctuationVectorizer, LengthTransformer, LengthOfSentenceTransformer, \
    SentimentTransformer, AverageWordLengthTransformer, CountPosVectorizer
from canary.argument_pipeline.model import Model

_ag = ArgumentDetector()


class AGTTransformer(object):

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[_ag.predict(y)] for y in x]


class ArgumentComponent(Model):

    def __init__(self, model_id=None, model_storage_location=None):
        self.model_id = "argument_component"

        super().__init__(model_id=self.model_id, model_storage_location=model_storage_location
                         )

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        train_data, test_data, train_targets, test_targets = load_essay_corpus()

        estimators = [
            ('logistic',
             LogisticRegression(max_iter=10000000)),
            ('SVM', SVC(
                random_state=0,
                gamma='auto',
                C=100000000,
                cache_size=100000,
                max_iter=-1,
                class_weight='balanced')
             ),
            ('rf', RandomForestClassifier(n_jobs=2, n_estimators=500))
        ]

        model = Pipeline([
            ('feats', FeatureUnion([
                ('tfidvectorizer',
                 TfidfVectorizer(ngram_range=(1, 8), tokenizer=Lemmatizer())),
                ('pos_tfid', TfidfPosVectorizer()),
                ('binary_length', LengthTransformer()),
                ('length_of_sentence', LengthOfSentenceTransformer()),
                ('average_word_length', AverageWordLengthTransformer()),
                ('punctuation_tfid_vectorizer', TfidfPunctuationVectorizer()),
                ('contain_claim', DiscourseMatcher(component='claim')),
                ('contain_premise', DiscourseMatcher(component='premise')),
                ('contain_major_claim', DiscourseMatcher(component='major_claim')),
                ('binary_first_person_i', FirstPersonIndicatorMatcher("I")),
                ('binary_first_person_me', FirstPersonIndicatorMatcher("me")),
                ('binary_first_person_my', FirstPersonIndicatorMatcher("my")),
                ('binary_first_person_myself', FirstPersonIndicatorMatcher("myself")),
                ('binary_first_person_mine', FirstPersonIndicatorMatcher("mine")),
                ('sentiment', SentimentTransformer()),
            ])),
            ('clf', StackingClassifier(
                estimators=estimators,
            ))
        ])

        return super(ArgumentComponent, self).train(pipeline_model=model,
                                                    train_data=train_data,
                                                    test_data=test_data,
                                                    train_targets=train_targets,
                                                    test_targets=test_targets)
