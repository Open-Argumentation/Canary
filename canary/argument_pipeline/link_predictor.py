from typing import Union

import nltk
import pandas
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler

import canary.utils
from canary.argument_pipeline.model import Model
from canary.preprocessing import Lemmatizer, PosLemmatizer, PosDistribution
from canary.preprocessing.transformers import DiscourseMatcher, SharedNouns

canary.preprocessing.nlp.nltk_download('punkt')
_nlp = canary.preprocessing.nlp.spacy_download()


class LinkPredictor(Model):

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            model_id = "link_predictor"

        __feats = [
            ("cv", TfidfVectorizer(tokenizer=PosLemmatizer(), max_features=500)),
            ('forward', DiscourseMatcher('forward', )),
            ('thesis', DiscourseMatcher('thesis', )),
            ('rebuttal', DiscourseMatcher('rebuttal', )),
            ('backward', DiscourseMatcher('backward', )),
        ]

        self.__dict_features = DictVectorizer()

        self.__arg1_text_features = FeatureUnion(__feats)

        self.__arg2_text_features = FeatureUnion(__feats)

        self.__scaler = RobustScaler(with_centering=False)

        super().__init__(model_id, model_storage_location)

    @staticmethod
    def default_train():
        from canary.corpora import load_essay_corpus
        return load_essay_corpus(purpose='link_prediction', train_split_size=0.8)

    def kfold_train(self):
        pass

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):

        if all(item is not None for item in [train_data, test_data, train_targets, test_targets]):
            train_dict, test_dict = self.prepare_dictionary_features(train_data, test_data)

            train_data = pandas.DataFrame(train_data)
            test_data = pandas.DataFrame(test_data)

            train_dict = self.__dict_features.fit_transform(train_dict)
            test_dict = self.__dict_features.transform(test_dict)

            canary.utils.logger.debug("Fitting...")
            self.__arg1_text_features.fit(train_data.arg1_cover_sen.tolist())
            arg1_feats_train = self.__arg1_text_features.transform(train_data.arg1_cover_sen.tolist())
            arg1_feats_test = self.__arg1_text_features.transform(test_data.arg1_cover_sen.tolist())

            self.__arg2_text_features.fit(train_data.arg2_cover_sen.tolist())
            arg2_feats_train = self.__arg2_text_features.transform(train_data.arg2_cover_sen.tolist())
            arg2_feats_test = self.__arg2_text_features.transform(test_data.arg2_cover_sen.tolist())

            train_data = hstack([train_dict, arg1_feats_train, arg2_feats_train])
            test_data = hstack([test_dict, arg1_feats_test, arg2_feats_test])

            canary.utils.logger.debug("Scaling...")

            self.__scaler.fit(train_data)
            train_data = self.__scaler.transform(train_data)
            test_data = self.__scaler.transform(test_data)

            canary.utils.logger.debug("Train...")

        if pipeline_model is None:
            pipeline_model = LogisticRegression(random_state=0, class_weight='balanced')

        return super().train(pipeline_model, train_data, test_data, train_targets, test_targets, save_on_finish, *args,
                             **kwargs)

    @staticmethod
    def prepare_dictionary_features(*data):
        canary.utils.logger.debug("Getting dictionary features")
        ret_tuple = ()
        # neg_word_transformer = WordSentimentCounter("neg")
        # pos_word_transformer = WordSentimentCounter("pos")
        pos_dist = PosDistribution()

        def get_features(feats):

            new_feats = feats.copy()
            shared_noun_counter = SharedNouns()
            for i, f in enumerate(new_feats):
                n_shared_nouns = shared_noun_counter.transform(f["arg1_component"], f["arg2_component"])

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
                    "shared_nouns": n_shared_nouns,
                    "both_in_conclusion": f["arg1_in_conclusion"] is True and f["arg2_in_conclusion"] is True,
                    "both_in_intro": f["arg1_in_intro"] is True and f["arg2_in_intro"] is True,
                    "n_para_components": f['n_para_components'],
                    "n_components_between_pair": abs(f["arg2_position"] - f["arg1_position"]),
                    "arg1_component_token_len": len(nltk.word_tokenize(f['arg1_component'])),
                    "arg2_component_token_len": len(nltk.word_tokenize(f['arg2_component'])),
                    "arg1_type": f["arg1_type"],
                    "arg2_type": f["arg2_type"],
                }

                arg1_posd = {"arg1_" + str(key): val for key, val in pos_dist(f['arg1_cover_sen']).items()}
                arg2_posd = {"arg2_" + str(key): val for key, val in pos_dist(f['arg2_cover_sen']).items()}

                features.update(arg1_posd)
                features.update(arg2_posd)

                new_feats[i] = features
            return new_feats

        for i, d in enumerate(data):
            canary.utils.logger.debug(f"{i + 1} / {len(data)}")
            ret_tuple = (*ret_tuple, get_features(d))

        return ret_tuple

    def get_relation_of_components(self, component_pairs, document):
        pass

    def predict(self, data, probability=False) -> Union[list, bool]:
        dict_feats = self.__dict_features.transform(self.prepare_dictionary_features([data])[0])

        arg1_feats = self.__arg1_text_features.transform([data['arg1_cover_sen']])
        arg2_feats = self.__arg2_text_features.transform([data['arg2_cover_sen']])

        combined_feats = hstack([arg1_feats, arg2_feats, dict_feats])
        combined_feats = self.__scaler.transform(combined_feats)

        return super().predict(combined_feats, probability)
