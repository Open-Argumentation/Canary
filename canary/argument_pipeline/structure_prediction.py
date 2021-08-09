from typing import Union

import pandas as pd
import spacy
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer, MaxAbsScaler

from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing.transformers import WordSentimentCounter, TfidfPosVectorizer, \
    EmbeddingTransformer, SentimentTransformer, DiscourseMatcher, AverageWordLengthTransformer, \
    LengthOfSentenceTransformer


class StructurePredictor(Model):

    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "structure_predictor"
        else:
            self.model_id = model_id

        super().__init__(model_id=self.model_id, model_storage_location=model_storage_location, load=load)

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=False, **kwargs):

        _train_data, _test_data, train_targets, test_targets = load_essay_corpus(purpose="relation_prediction")

        pd_train = pd.DataFrame(_train_data)
        pd_test = pd.DataFrame(_test_data)

        text_feats = FeatureUnion([
            ("tfidf_unigrams",
             TfidfVectorizer(ngram_range=(1, 2), lowercase=False, )),
            ('posv', TfidfPosVectorizer(ngrams=2)),
            ("sentiment_pos", SentimentTransformer("pos")),
            ("sentiment_neg", SentimentTransformer("neg")),
            ("sentiment_neu", SentimentTransformer("neu")),

        ])

        cover_feats = FeatureUnion([
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

        cover_feats.fit(pd_train.arg1_covering_sentence.tolist())
        c1 = cover_feats.transform(pd_train.arg1_covering_sentence.tolist())
        c2 = cover_feats.transform(pd_test.arg1_covering_sentence.tolist())

        cover_feats.fit(pd_train.arg2_covering_sentence.tolist())
        c3 = cover_feats.transform(pd_train.arg2_covering_sentence.tolist())
        c4 = cover_feats.transform(pd_test.arg2_covering_sentence.tolist())

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

        combined_features_train = hstack([t1, t3, f1, c1, c3])
        combined_features_test = hstack([t2, t4, f2, c2, c4])

        scaler = MaxAbsScaler().fit(combined_features_train)
        combined_features_train = scaler.transform(combined_features_train)
        combined_features_test = scaler.transform(combined_features_test)

        combined_features_train = hstack([combined_features_train, o1, o3, ])
        combined_features_test = hstack([combined_features_test, o2, o4, ])

        if pipeline_model is None:
            sgd = SGDClassifier(
                class_weight='balanced',
                random_state=0,
                loss='modified_huber',
                alpha=0.000001
            )

            model = sgd
        else:
            model = pipeline_model

        super(StructurePredictor, self).train(
            pipeline_model=model,
            train_data=combined_features_train,
            test_data=combined_features_test,
            train_targets=train_targets,
            test_targets=test_targets,
        )

        self.save(model_data={
            "model_id": self.model_id,
            "model": self.model,
            "metrics": self.metrics
        })

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
                })

            return features

        train_data = get_features(train)
        test_data = get_features(test)

        return train_data, test_data

    def predict(self, data: Union[list, str], probability=False) -> Union[list, bool]:
        return None


def binary_neg_present(sen):
    return WordSentimentCounter("neg").transform([sen])[0][0] > 0
