from typing import Union

import nltk
import pandas
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler

import canary.utils
from canary.argument_pipeline.model import Model
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import DiscourseMatcher

canary.preprocessing.nlp.nltk_download('punkt')
nlp = canary.preprocessing.nlp.spacy_download()


class LinkPredictor(Model):

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            model_id = "link_predictor"

        self.__dict_features = FeatureHasher()

        self.__arg1_text_features = FeatureUnion([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
            ("support", DiscourseMatcher("support")),
            ("conflict", DiscourseMatcher("conflict")),
            ('forward', DiscourseMatcher('forward')),
            ('thesis', DiscourseMatcher('thesis')),
        ])

        self.__arg2_text_features = FeatureUnion([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
            ("support", DiscourseMatcher("support")),
            ("conflict", DiscourseMatcher("conflict")),
            ('forward', DiscourseMatcher('forward')),
            ('thesis', DiscourseMatcher('thesis')),
        ])

        self.__scaler = MaxAbsScaler()

        super().__init__(model_id, model_storage_location)

    @staticmethod
    def default_train():
        from canary.corpora import load_essay_corpus
        return load_essay_corpus(purpose='link_prediction', train_split_size=0.6)

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):
        if all(item is not None for item in [train_data, test_data, train_targets, test_targets]):
            train_dict, test_dict = self.prepare_dictionary_features(train_data, test_data)

            train_data = pandas.DataFrame(train_data)
            test_data = pandas.DataFrame(test_data)

            train_dict = self.__dict_features.fit_transform(train_dict)
            test_dict = self.__dict_features.transform(test_dict)

            self.__arg1_text_features.fit(train_data.arg1_cover_sen.tolist())
            arg1_feats_train = self.__arg1_text_features.transform(train_data.arg1_cover_sen.tolist())
            arg1_feats_test = self.__arg1_text_features.transform(test_data.arg1_cover_sen.tolist())

            self.__arg2_text_features.fit(train_data.arg2_cover_sen.tolist())
            arg2_feats_train = self.__arg2_text_features.transform(train_data.arg2_cover_sen.tolist())
            arg2_feats_test = self.__arg2_text_features.transform(test_data.arg2_cover_sen.tolist())

            train_data = hstack([train_dict, arg1_feats_train, arg2_feats_train])
            test_data = hstack([test_dict, arg1_feats_test, arg2_feats_test])

            self.__scaler.fit(train_data)
            train_data = self.__scaler.transform(train_data)
            test_data = self.__scaler.transform(test_data)

        if pipeline_model is None:
            pipeline_model = LogisticRegression(class_weight='balanced', random_state=0, warm_start=True, solver='sag')

        return super().train(pipeline_model, train_data, test_data, train_targets, test_targets, save_on_finish, *args,
                             **kwargs)

    @staticmethod
    def prepare_dictionary_features(*data):
        canary.utils.logger.debug("Getting dictionary features")

        ret_tuple = ()

        def get_features(feats):
            new_feats = feats.copy()
            for i, f in enumerate(new_feats):
                canary.utils.logger.debug(f"{i}/{len(new_feats)}...")
                # arg1 = nlp(f["arg1_component"])
                # arg2 = nlp(f['arg2_component'])

                features = {
                    "arg1_component": f["arg1_component"],
                    "arg2_component": f['arg2_component'],
                    'arg1_type': f['arg1_type'],
                    'arg2_type': f['arg2_type'],
                    "arg1_cover_sen_token_len": len(nltk.word_tokenize(f['arg1_cover_sen'])),
                    "arg2_cover_sen_token_len": len(nltk.word_tokenize(f['arg2_cover_sen'])),
                    "arg1_component_token_len": len(nltk.word_tokenize(f['arg1_component'])),
                    "arg2_component_token_len": len(nltk.word_tokenize(f['arg2_component'])),
                }

                new_feats[i] = features
            return new_feats

        for d in data:
            ret_tuple = (*ret_tuple, get_features(d))

        return ret_tuple

    def predict(self, data, probability=False) -> Union[list, bool]:
        raise NotImplementedError()
