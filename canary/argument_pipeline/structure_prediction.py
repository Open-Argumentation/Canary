from typing import Union

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer, RobustScaler

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

        self.__cover_feats = FeatureUnion([
            ("support", DiscourseMatcher("support")),
            ("conflict", DiscourseMatcher("conflict")),
            ('forward', DiscourseMatcher('forward')),
            ('thesis', DiscourseMatcher('thesis')),
            ("average_word_length", AverageWordLengthTransformer()),
            ('length_of_sentence', LengthOfSentenceTransformer()),
            ('el', EmbeddingTransformer()),
            ("wc1", WordSentimentCounter("neu")),
            ("w2", WordSentimentCounter("pos")),
            ("wc3", WordSentimentCounter("neg")),
        ])

        self.__text_feats = FeatureUnion([
            ("tfidf_unigrams",
             TfidfVectorizer(ngram_range=(1, 2), lowercase=False, max_features=1000)),
            ('posv', TfidfPosVectorizer()),
            ("sentiment_pos", SentimentTransformer("pos")),
            ("sentiment_neg", SentimentTransformer("neg")),
            ("sentiment_neu", SentimentTransformer("neu")),
        ])

        self.__feature_hasher = FeatureHasher()
        self.__ohe = LabelBinarizer()
        self.__scaler = RobustScaler(with_centering=False)

        super().__init__(model_id=model_id, model_storage_location=model_storage_location)

    @staticmethod
    def default_train():
        canary.logger.debug("Getting default data")
        return load_essay_corpus(purpose="relation_prediction")

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, **kwargs):
        if all(item is not None for item in [train_data, test_data, train_targets, test_targets]):
            pd_train = pd.DataFrame(train_data)
            pd_test = pd.DataFrame(test_data)

            canary.logger.debug("Getting dictionary features")
            _train_data_dict, _test_data_dict = self.prepare_dictionary_features(train_data, test_data)

            canary.logger.debug("Fitting features")
            self.__cover_feats.fit(pd_train.arg1_covering_sentence.tolist())
            c1 = self.__cover_feats.transform(pd_train.arg1_covering_sentence.tolist())
            c2 = self.__cover_feats.transform(pd_test.arg1_covering_sentence.tolist())

            self.__cover_feats.fit(pd_train.arg2_covering_sentence.tolist())
            c3 = self.__cover_feats.transform(pd_train.arg2_covering_sentence.tolist())
            c4 = self.__cover_feats.transform(pd_test.arg2_covering_sentence.tolist())

            self.__text_feats.fit(pd_train.arg1_text.tolist())
            t1 = self.__text_feats.transform(pd_train.arg1_text.tolist())
            t2 = self.__text_feats.transform(pd_test.arg1_text.tolist())

            self.__text_feats.fit(pd_train.arg2_text.tolist())
            t3 = self.__text_feats.transform(pd_train.arg2_text.tolist())
            t4 = self.__text_feats.transform(pd_test.arg2_text.tolist())

            self.__feature_hasher.fit(_train_data_dict)
            f1 = self.__feature_hasher.transform(_train_data_dict)
            f2 = self.__feature_hasher.transform(_test_data_dict)

            o1 = self.__ohe.fit_transform(pd_train.arg1_type.tolist())
            o2 = self.__ohe.transform(pd_test.arg1_type.tolist())

            o3 = self.__ohe.fit_transform(pd_train.arg2_type.tolist())
            o4 = self.__ohe.transform(pd_test.arg2_type.tolist())

            combined_features_train = hstack([t1, t3, f1, c1, c3])
            combined_features_test = hstack([t2, t4, f2, c2, c4])

            canary.logger.debug("Scale features")
            self.__scaler.fit(combined_features_train)
            combined_features_train = self.__scaler.transform(combined_features_train)
            combined_features_test = self.__scaler.transform(combined_features_test)

            train_data = hstack([combined_features_train, o1, o3, ])
            test_data = hstack([combined_features_test, o2, o4, ])

        if pipeline_model is None:
            pipeline_model = SGDClassifier(
                class_weight='balanced',
                random_state=0,
                loss='modified_huber',
                alpha=0.000001
            )

        super(StructurePredictor, self).train(
            pipeline_model=pipeline_model,
            train_data=train_data,
            test_data=test_data,
            train_targets=train_targets,
            test_targets=test_targets,
            save_on_finish=save_on_finish
        )

    @staticmethod
    def prepare_dictionary_features(train, test):
        nlp = canary.utils.spacy_download()

        def get_features(data):
            features = []

            def get_feats(d):
                sent1 = nlp(d["arg1_covering_sentence"])
                sent2 = nlp(d["arg2_covering_sentence"])
                return {
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
                    "neg_present_arg1": binary_neg_present(sent1.text),
                    "neg_present_arg2": binary_neg_present(sent2.text),
                }

            for d in data:
                features.append(get_feats(d))

            return features

        train_data = get_features(train)
        test_data = get_features(test)

        return train_data, test_data

    def predict(self, data: Union[list, str], probability=False) -> Union[list, bool]:
        raise NotImplementedError()


def binary_neg_present(sen):
    return WordSentimentCounter("neg").transform([sen])[0][0] > 0
