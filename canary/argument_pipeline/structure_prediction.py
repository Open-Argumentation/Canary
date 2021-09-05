import pandas
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import MaxAbsScaler

import canary.utils
from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing.transformers import WordSentimentCounter, TfidfPosVectorizer, \
    EmbeddingTransformer, SentimentTransformer, DiscourseMatcher, AverageWordLengthTransformer, \
    LengthOfSentenceTransformer


class StructurePredictor(Model):

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            model_id = "structure_predictor"

        super().__init__(model_id=model_id, model_storage_location=model_storage_location)

    @staticmethod
    def default_train():
        canary.utils.logger.debug("Getting default data")
        return load_essay_corpus(purpose="relation_prediction")

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, **kwargs):

        if pipeline_model is None:
            pipeline_model = make_pipeline(
                StructureFeatures(),
                MaxAbsScaler(),
                SGDClassifier(
                    class_weight='balanced',
                    random_state=0,
                    loss='modified_huber',
                    alpha=0.000001
                ))

        super(StructurePredictor, self).train(
            pipeline_model=pipeline_model,
            train_data=train_data,
            test_data=test_data,
            train_targets=train_targets,
            test_targets=test_targets,
            save_on_finish=save_on_finish
        )


class StructureFeatures(TransformerMixin, BaseEstimator):
    cover_features: list = [
        DiscourseMatcher("support"),
        DiscourseMatcher("conflict"),
        DiscourseMatcher('forward'),
        DiscourseMatcher('thesis'),
        AverageWordLengthTransformer(),
        LengthOfSentenceTransformer(),
        EmbeddingTransformer(),
        WordSentimentCounter("neu"),
        WordSentimentCounter("pos"),
        WordSentimentCounter("neg")
    ]

    text_features: list = [
        TfidfVectorizer(ngram_range=(1, 2), lowercase=False, max_features=1000),
        TfidfPosVectorizer(),
        SentimentTransformer("pos"),
        SentimentTransformer("neg"),
        SentimentTransformer("neu"),
    ]

    def __init__(self):
        self.__arg1_cover_features = make_union(*StructureFeatures.cover_features)
        self.__arg2_cover_features = make_union(*StructureFeatures.cover_features)

        self.__arg1_text_features = make_union(*StructureFeatures.text_features)
        self.__arg2_text_features = make_union(*StructureFeatures.text_features)

        self.__dictionary_features = FeatureHasher()

    def fit(self, x, y=None):
        canary.utils.logger.debug("fitting...")

        self.__dictionary_features.fit(self.prepare_dictionary_features(x))
        x = pandas.DataFrame(x)

        self.__arg1_cover_features.fit(x.arg1_covering_sentence.tolist())
        self.__arg2_cover_features.fit(x.arg2_covering_sentence.tolist())

        self.__arg1_text_features.fit(x.arg1_text.tolist())
        self.__arg2_text_features.fit(x.arg2_text.tolist())
        return self

    def transform(self, x):
        canary.utils.logger.debug("transforming...")
        dictionary_features = self.__dictionary_features.transform(x)

        x = pandas.DataFrame(x)
        arg1_cover_features = self.__arg1_cover_features.transform(x.arg1_covering_sentence)
        arg2_cover_features = self.__arg2_cover_features.transform(x.arg2_covering_sentence)

        arg1_text_features = self.__arg1_text_features.transform(x.arg2_text)
        arg2_text_features = self.__arg2_text_features.transform(x.arg2_text)

        return hstack(
            [
                dictionary_features,
                arg1_cover_features,
                arg2_cover_features,
                arg1_text_features,
                arg2_text_features
            ]
        )

    @staticmethod
    def binary_neg_present(sen):
        return WordSentimentCounter("neg").transform([sen])[0][0] > 0

    @staticmethod
    def prepare_dictionary_features(data):
        canary.utils.logger.debug("Getting dictionary features.")
        nlp = canary.preprocessing.nlp.spacy_download()

        def get_features(f):
            new_feats = f.copy()
            for t, d in enumerate(new_feats):
                sent1 = nlp(d["arg1_covering_sentence"])
                sent2 = nlp(d["arg2_covering_sentence"])
                new_feats[t] = {
                    "arg1_type": d["arg1_type"],
                    "arg2_type": d["arg2_type"],
                    "arg1_start": d["arg1_start"],
                    "arg2_start": d["arg2_start"],
                    "arg1_end": d["arg1_end"],
                    "arg2_end": d["arg2_end"],
                    'arg1_preceding_tokens': d['arg1_preceding_tokens'],
                    "arg1_following_tokens": d["arg1_following_tokens"],
                    'arg2_preceding_tokens': d['arg2_preceding_tokens'],
                    "arg2_following_tokens": d["arg2_following_tokens"],
                    "sentence_similarity_norm": sent1.similarity(sent2),
                    "n_preceding_components": d["n_preceding_components"],
                    "n_following_components": d["n_following_components"],
                    "n_attack_components": d["n_attack_components"],
                    "n_support_components": d["n_support_components"],
                    "arg1_cover_ents": len(sent1.ents),
                    "arg2_cover_ents": len(sent2.ents),
                    "neg_present_arg1": StructureFeatures.binary_neg_present(sent1.text),
                    "neg_present_arg2": StructureFeatures.binary_neg_present(sent2.text),
                }
            return new_feats

        return get_features(data)
