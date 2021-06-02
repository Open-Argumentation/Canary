from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing import Lemmatizer, Preprocessor
from canary.preprocessing.transformers import TfidfPosVectorizer, DiscourseMatcher, \
    FirstPersonIndicatorMatcher, TfidfPunctuationVectorizer, LengthTransformer, LengthOfSentenceTransformer, \
    SentimentTransformer, AverageWordLengthTransformer, UniqueWordsTransformer


class ArgumentComponent(Model):

    def __init__(self, model_id=None, model_storage_location=None, load=True):

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
                loss='log',
                random_state=0,
                warm_start=True,
                early_stopping=True,
            )),
            ('SVC', SVC(
                random_state=0,
                C=9000000,
                max_iter=-1,
            )
             ),
            ('KNeighborsClassifier', KNeighborsClassifier(leaf_size=60, n_jobs=2)),
            ("AdaBoostClassifier", AdaBoostClassifier(n_estimators=100, random_state=0)),
        ]

        model = Pipeline([
            ('feats', FeatureUnion([
                ('tfidvectorizer',
                 TfidfVectorizer(ngram_range=(1, 3), tokenizer=Lemmatizer(), lowercase=False)),
                ('pos_tfid', TfidfPosVectorizer(stop_words=Preprocessor().stopwords)),
                ('binary_length', LengthTransformer(word_length=8)),
                ('ddddd', LengthTransformer(word_length=12)),
                ('bl4', LengthTransformer(word_length=4)),
                ('punctuation_tfid_vectorizer', TfidfPunctuationVectorizer()),
                ('length_of_sentence', LengthOfSentenceTransformer()),
                ('average_word_length', AverageWordLengthTransformer()),
                ('contain_claim', DiscourseMatcher(component='claim')),
                ('contain_premise', DiscourseMatcher(component='premise')),
                ('contain_major_claim', DiscourseMatcher(component='major_claim')),
                ('support', DiscourseMatcher(component='support')),
                ('conflict', DiscourseMatcher(component='conflict')),
                ('obligation', DiscourseMatcher(component='obligation')),
                ('recommendation', DiscourseMatcher(component='recommendation')),
                ('option', DiscourseMatcher(component='option')),
                ('intention', DiscourseMatcher(component='intention')),
                ('binary_first_person_i', FirstPersonIndicatorMatcher("I")),
                ('binary_first_person_me', FirstPersonIndicatorMatcher("me")),
                ('binary_first_person_my', FirstPersonIndicatorMatcher("my")),
                ('binary_first_person_myself', FirstPersonIndicatorMatcher("myself")),
                ('binary_first_person_mine', FirstPersonIndicatorMatcher("mine")),
                ('sentiment', SentimentTransformer()),
                ("UniqueWordsTransformer", UniqueWordsTransformer())
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
