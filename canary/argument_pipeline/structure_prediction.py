import pandas as pd
import spacy
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer, MaxAbsScaler
from sklearn.svm import SVC, LinearSVC

from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import WordSentimentCounter, TfidfPosVectorizer, \
    EmbeddingTransformer, SentimentTransformer


class StructurePredictor(Model):

    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "structure_predictor"

        super().__init__(model_id=self.model_id, model_storage_location=model_storage_location, load=load)

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        _train_data, _test_data, train_targets, test_targets = load_essay_corpus(purpose="relation_prediction")

        pd_train = pd.DataFrame(_train_data)
        pd_test = pd.DataFrame(_test_data)

        text_feats = FeatureUnion([
            ("tfidf_pos",
             TfidfVectorizer(ngram_range=(1, 1), lowercase=False, tokenizer=Lemmatizer(), )),
            ('posv', TfidfPosVectorizer()),
            ("sentiment_pos", SentimentTransformer("pos")),
            ("sentiment_neg", SentimentTransformer("neg")),
            ("sentiment_neu", SentimentTransformer("neu")),
            ('el', EmbeddingTransformer())
        ])

        num_feats = FeatureUnion([
            ("wc1", WordSentimentCounter("neu")),
            ("w2", WordSentimentCounter("pos")),
            ("wc3", WordSentimentCounter("neg")),
        ])

        num_feats.fit(pd_train.arg1_text.tolist())
        n1 = num_feats.transform(pd_train.arg1_text.tolist())
        n2 = num_feats.transform(pd_test.arg1_text.tolist())

        num_feats.fit(pd_train.arg2_text.tolist())
        n3 = num_feats.transform(pd_train.arg2_text.tolist())
        n4 = num_feats.transform(pd_test.arg2_text.tolist())

        text_feats.fit(pd_train.arg1_text.tolist())
        t1 = text_feats.transform(pd_train.arg1_text.tolist())
        t2 = text_feats.transform(pd_test.arg1_text.tolist())

        text_feats.fit(pd_train.arg2_text.tolist())
        t3 = text_feats.transform(pd_train.arg2_text.tolist())
        t4 = text_feats.transform(pd_test.arg2_text.tolist())

        _train_data_dict, _test_data_dict = self.prepare_dictionary_features(_train_data, _test_data)

        f = FeatureHasher().fit(_train_data_dict)
        f1 = f.transform(_train_data_dict)
        f2 = f.transform(_test_data_dict)

        ohe = LabelBinarizer()
        o1 = ohe.fit_transform(pd_train.arg1_type.tolist())
        o2 = ohe.transform(pd_test.arg1_type.tolist())

        o3 = ohe.fit_transform(pd_train.arg2_type.tolist())
        o4 = ohe.transform(pd_test.arg2_type.tolist())

        combined_features_train = hstack([t1, t3, f1, n1, n3])
        combined_features_test = hstack([t2, t4, f2, n2, n4])

        scaler = MaxAbsScaler().fit(combined_features_train)
        combined_features_train = scaler.transform(combined_features_train)
        combined_features_test = scaler.transform(combined_features_test)

        combined_features_train = hstack([combined_features_train, o1, o3, ])
        combined_features_test = hstack([combined_features_test, o2, o4, ])

        sgd = SGDClassifier(
            class_weight='balanced',
            random_state=0,
            warm_start=True,
            loss='modified_huber',
            verbose=1
        )

        knn = KNeighborsClassifier(n_neighbors=3, p=2, )
        svc = SVC(kernel='linear', probability=True, random_state=0, gamma='auto')
        lsvc = LinearSVC(random_state=0)
        estimators = [
            ("sgd", sgd),
            ('knn', knn),
            ('svc', svc)
        ]
        mb = OneVsRestClassifier(MultinomialNB(
            fit_prior=True, class_prior=None))
        lg = LogisticRegression(random_state=0, class_weight="balanced")
        model = lg

        super(StructurePredictor, self).train(
            pipeline_model=model,
            train_data=combined_features_train,
            test_data=combined_features_test,
            train_targets=train_targets,
            test_targets=test_targets
        )

    def prepare_dictionary_features(self, train, test):
        nlp = spacy.load('en_core_web_lg')

        def get_features(data):
            features = []

            for d in data:
                sent1 = nlp(d["arg1_covering_sentence"])
                sent2 = nlp(d["arg2_covering_sentence"])

                features.append({
                    "arg1_start": d["arg1_start"],
                    "arg2_start": d["arg2_start"],
                    "arg1_end": d["arg1_end"],
                    "arg2_end": d["arg2_end"],
                    "sentence_similarity_norm": sent1.similarity(sent2),
                    "n_preceding_components": d["n_preceding_components"],
                    "n_following_components": d["n_following_components"],
                    "n_attack_components": d["n_attack_components"],
                    "n_support_components": d["n_support_components"]
                })

            return features

        train_data = get_features(train)
        test_data = get_features(test)

        return train_data, test_data


def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),

    }

    return features


def sent2features(sent):
    import nltk
    sent = nltk.word_tokenize(sent)
    return [word2features(sent, i) for i in range(len(sent))]
