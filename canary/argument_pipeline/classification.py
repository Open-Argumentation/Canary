from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from canary.argument_pipeline.model import Model
from canary.corpora import load_ukp_sentential_argument_detection_corpus
from canary.preprocessing import Preprocessor

from canary.preprocessing.transformers import CountPosVectorizer, DiscourseMatcher, CountPunctuationVectorizer, \
    LengthOfSentenceTransformer


class ArgumentDetector(Model):

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            self.model_id = "argument_detector"
        super().__init__(model_id=self.model_id, model_storage_location=model_storage_location,
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
                ('pos_tagger', CountPosVectorizer()),
                ('bow',
                 CountVectorizer(
                     ngram_range=(1, 6),
                     stop_words=Preprocessor().stopwords)
                 ),
                ('length', LengthOfSentenceTransformer()),
                ("wp", CountVectorizer(ngram_range=(2, 2))),
                ("discourse", DiscourseMatcher()),
                ("punctuation", CountPunctuationVectorizer()),
            ])),
            ('clf', StackingClassifier(
                estimators=[
                    ('a', ComplementNB()),
                    ('b', RandomForestClassifier()),
                    ('c', SGDClassifier()),
                    ('d', KNeighborsClassifier(
                        n_neighbors=50,
                        metric='euclidean'
                    )),
                ],
                final_estimator=LogisticRegression(random_state=0)))
        ])
        return super(ArgumentDetector, self).train(pipeline_model=model,
                                                   train_data=train_data,
                                                   test_data=test_data,
                                                   train_targets=train_targets,
                                                   test_targets=test_targets)
