from sklearn.linear_model import SGDClassifier, LogisticRegression

from canary.preprocessing import Lemmatizer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from canary.corpora import load_essay_corpus
from canary.preprocessing.transformers import TfidfPosVectorizer, DiscourseMatcher, \
    FirstPersonIndicatorMatcher, TfidfPunctuationVectorizer, LengthTransformer, LengthOfSentenceTransformer, \
    SentimentTransformer, AverageWordLengthTransformer
from canary.argument_pipeline.model import Model


class ArgumentComponent(Model):

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            self.model_id = "argument_component"
        else:
            self.model_id = model_id

        super().__init__(model_id=self.model_id, model_storage_location=model_storage_location
                         )

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        train_data, test_data, train_targets, test_targets = load_essay_corpus()

        estimators = [
            ('da', SGDClassifier(loss='log',
                                 random_state=0,
                                 warm_start=True,
                                 early_stopping=True,
                                 tol=1e-5
                                 )),
            ('SVM',
             SVC(
                 random_state=0,
                 C=9000000000,
                 max_iter=-1,
             )
             ),
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

        super(ArgumentComponent, self).train(pipeline_model=model,
                                             train_data=train_data,
                                             test_data=test_data,
                                             train_targets=train_targets,
                                             test_targets=test_targets)
