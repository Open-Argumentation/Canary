from typing import Union

import nltk
import pandas
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer

import canary.utils
from canary.argument_pipeline.model import Model
from canary.preprocessing import Lemmatizer, PosDistribution
from canary.preprocessing.transformers import DiscourseMatcher, SharedNouns

canary.preprocessing.nlp.nltk_download('punkt')
_nlp = canary.preprocessing.nlp.spacy_download()


class LinkPredictor(Model):

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            model_id = "link_predictor"

        super().__init__(model_id, model_storage_location)

    @staticmethod
    def default_train():
        from canary.corpora import load_essay_corpus

        train_data, test_data, train_targets, test_targets = load_essay_corpus(purpose='link_prediction',
                                                                               train_split_size=0.8)
        from imblearn.under_sampling import RandomUnderSampler
        canary.utils.logger.debug("Resample")
        ros = RandomUnderSampler(random_state=0)
        train_data, train_targets = ros.fit_resample(pandas.DataFrame(train_data), pandas.DataFrame(train_targets))

        return list(train_data.to_dict("index").values()), test_data, train_targets[0].tolist(), test_targets

    def stratified_k_fold_train(self, pipeline_model=None, train_data=None, train_targets=None,
                                save_on_finish=True, n_splits=2, *args, **kwargs):

        train_data, test_data, train_targets, test_targets = self.default_train()
        train_data = train_data + test_data
        train_targets = train_targets + test_targets

        del test_data
        del test_targets

        if pipeline_model is None:
            pipeline_model = make_pipeline(
                LinkFeatures(),
                Normalizer(),
                RandomForestClassifier(n_estimators=300, random_state=0, min_samples_leaf=4,
                                       max_depth=10)
            )

        return super(LinkPredictor, self).stratified_k_fold_train(pipeline_model, train_data, train_targets,
                                                                  save_on_finish, n_splits)

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):

        if pipeline_model is None:
            pipeline_model = make_pipeline(
                LinkFeatures(),
                Normalizer(),
                RandomForestClassifier(n_estimators=300, random_state=0, min_samples_leaf=4,
                                       max_depth=10)
            )

        return super().train(pipeline_model, train_data, test_data, train_targets, test_targets, save_on_finish, *args,
                             **kwargs)

    def predict(self, data, probability=False) -> Union[list, bool]:
        return super().predict(data, probability)[0]


class LinkFeatures(TransformerMixin, BaseEstimator):
    feats: list = [
        DiscourseMatcher('forward'),
        DiscourseMatcher('thesis'),
        DiscourseMatcher('rebuttal'),
        DiscourseMatcher('backward'),
    ]

    def __init__(self):

        self.__nom_dict_features = DictVectorizer()

        self.__numeric_dict_features = DictVectorizer()

        self.__arg1_cover_features = make_union(*LinkFeatures.feats.copy())

        self.__arg2_cover_features = make_union(*LinkFeatures.feats.copy())

    def fit(self, x, y=None):
        """

        :param x:
        :param y: ignored.
        :return:
        """

        num_dict = self.prepare_numeric_feats(x)
        self.__numeric_dict_features.fit(num_dict)
        self.__nom_dict_features.fit(self.prepare_dictionary_features(x))

        x = pandas.DataFrame(x)

        self.__arg1_cover_features.fit(x.arg1_cover_sen.tolist())
        self.__arg2_cover_features.fit(x.arg2_cover_sen.tolist())

        return self

    def transform(self, x):
        dict_feats = self.__nom_dict_features.transform(self.prepare_dictionary_features(x))
        num_dict_feats = self.__numeric_dict_features.transform(self.prepare_numeric_feats(x))

        x = pandas.DataFrame(x)

        arg1_cover_feats = self.__arg1_cover_features.transform(x.arg1_cover_sen)
        arg2_cover_feats = self.__arg2_cover_features.transform(x.arg2_cover_sen)

        return hstack([dict_feats, num_dict_feats, arg1_cover_feats, arg2_cover_feats, ])

    @staticmethod
    def prepare_dictionary_features(data):
        canary.utils.logger.debug("Getting dictionary features")

        def get_features(feats):
            new_feats = feats.copy()
            for t, f in enumerate(new_feats):
                features = {
                    "source_before_target": f["source_before_target"],
                    "arg1_first_in_paragraph": f["arg1_first_in_paragraph"],
                    "arg1_last_in_paragraph": f["arg1_last_in_paragraph"],
                    "arg2_first_in_paragraph": f["arg2_first_in_paragraph"],
                    "arg2_last_in_paragraph": f["arg2_last_in_paragraph"],
                    "arg1_in_intro": f["arg1_in_intro"],
                    "arg1_in_conclusion": f["arg1_in_conclusion"],
                    "arg2_in_intro": f['arg2_in_intro'],
                    "arg2_in_conclusion": f["arg2_in_conclusion"],
                    "arg1_and_arg2_in_same_sentence": f["arg1_and_arg2_in_same_sentence"],
                    "both_in_conclusion": f["arg1_in_conclusion"] is True and f["arg2_in_conclusion"] is True,
                    "both_in_intro": f["arg1_in_intro"] is True and f["arg2_in_intro"] is True,
                }

                new_feats[t] = features
            return new_feats

        return get_features(data)

    @staticmethod
    def prepare_numeric_feats(data):
        shared_noun_counter = SharedNouns()
        canary.utils.logger.debug("Getting numeric features")

        def get_features(feats):
            pos_dist = PosDistribution()

            new_feats = feats.copy()
            for t, f in enumerate(new_feats):
                n_shared_nouns = shared_noun_counter.transform(f["arg1_component"], f["arg2_component"])

                features = {
                    "n_para_components": f['n_para_components'],
                    "n_components_between_pair": abs(f["arg2_position"] - f["arg1_position"]),
                    "arg1_component_token_len": len(nltk.word_tokenize(f['arg1_component'])),
                    "arg2_component_token_len": len(nltk.word_tokenize(f['arg2_component'])),
                    "arg1_cover_sen_token_len": len(nltk.word_tokenize(f['arg1_cover_sen'])),
                    "arg2_cover_sen_token_len": len(nltk.word_tokenize(f['arg2_cover_sen'])),
                    "arg1_type": f["arg1_type"],
                    "arg2_type": f["arg2_type"],
                    "shared_nouns": n_shared_nouns,
                }

                arg1_posd = {"arg1_" + str(key): val for key, val in pos_dist(f['arg1_cover_sen']).items()}
                arg2_posd = {"arg2_" + str(key): val for key, val in pos_dist(f['arg2_cover_sen']).items()}

                features.update(arg1_posd)
                features.update(arg2_posd)

                new_feats[t] = features
            return new_feats

        return get_features(data)
