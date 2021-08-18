from typing import Union

import pandas
from nltk.tree import Tree
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler

import canary
import canary.utils
from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import DiscourseMatcher, FirstPersonIndicatorMatcher


class ArgumentComponent(Model):
    """
    Detects argumentative components from natural language
    """

    def __init__(self, model_id: str = None, model_storage_location=None):
        """
        :param model_id: the ID of the model
        :param model_storage_location: where the model should be stored
        """

        if model_id is None:
            model_id = "argument_component"

        self.__feats = FeatureUnion([
            ('tfidvectorizer',
             TfidfVectorizer(ngram_range=(1, 3), tokenizer=Lemmatizer(), lowercase=False, max_features=1000)),
            ('forward', DiscourseMatcher('forward', lemmatize=True)),
            ('thesis', DiscourseMatcher('thesis', lemmatize=True)),
            ('rebuttal', DiscourseMatcher('rebuttal', lemmatize=True)),
            ('backward', DiscourseMatcher('backward', lemmatize=True)),
            ("first_i", FirstPersonIndicatorMatcher("I")),
            ("first_me", FirstPersonIndicatorMatcher("me")),
            ("first_mine", FirstPersonIndicatorMatcher("mine")),
            ("first_myself", FirstPersonIndicatorMatcher("myself")),
            ("first_my", FirstPersonIndicatorMatcher("my")),
        ])

        self.__dict_feats = FeatureHasher()
        self.__scaler = RobustScaler(with_centering=False, unit_variance=False)

        super().__init__(
            model_id=model_id,
            model_storage_location=model_storage_location,
        )

    @staticmethod
    def default_train():
        # get training and test data
        train_data, test_data, train_targets, test_targets = load_essay_corpus(
            purpose="component_prediction",
            train_split_size=0.6
        )

        return train_data, test_data, train_targets, test_targets

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):

        train_feats = None
        test_feats = None

        # The below functionality requires the following is not None.
        if all(item is not None for item in [train_data, test_data, train_targets, test_targets]):
            canary.logger.debug("Getting dictionary features")
            train_dict, test_dict = self.prepare_dictionary_features(train_data, test_data)

            train_data = pandas.DataFrame(train_data)
            test_data = pandas.DataFrame(test_data)

            # Fit training data
            canary.logger.debug("fitting features")
            self.__dict_feats.fit(train_dict)
            self.__feats.fit(train_data.cover_sentence.tolist())

            train_feats = self.__feats.transform(train_data.cover_sentence.tolist())
            test_feats = self.__feats.transform(test_data.cover_sentence.tolist())

            train_dict_feats = self.__dict_feats.transform(train_dict)
            test_dict_feats = self.__dict_feats.transform(test_dict)

            # combine feats into one vector
            train_feats = hstack([train_feats, train_dict_feats])
            test_feats = hstack([test_feats, test_dict_feats])

            # scale
            self.__scaler.fit(train_feats)
            train_feats = self.__scaler.transform(train_feats)
            test_feats = self.__scaler.transform(test_feats)

        # If the pipeline model is none, use this algorithm
        if pipeline_model is None:
            pipeline_model = LogisticRegression(
                class_weight='balanced',
                warm_start=True,
                random_state=0,
                max_iter=2000,
                solver="newton-cg"
            )

        super(ArgumentComponent, self).train(
            pipeline_model=pipeline_model,
            train_data=train_feats,
            test_data=test_feats,
            train_targets=train_targets,
            test_targets=test_targets,
            save_on_finish=save_on_finish
        )

    @staticmethod
    def prepare_dictionary_features(*feats):
        ret_tuple = ()
        nlp = canary.utils.spacy_download()

        def get_features(data):
            canary.logger.debug("getting dictionary features.")
            features = []

            for d in data:
                sen = nlp(d['cover_sentence'])
                cover_sen_parse_tree = Tree.fromstring(list(sen.sents)[0]._.parse_string)

                items = {
                    "parse_tree_height": cover_sen_parse_tree.height(),
                    'len_paragraph': d['len_paragraph'],
                    "len_component": d['len_component'],
                    "len_cover_sen": d['len_cover_sen'],
                    'is_in_intro': d['is_in_intro'],
                    'is_in_conclusion': d['is_in_conclusion'],
                    "n_following_components": d["n_following_components"],
                    "n_preceding_components": d["n_preceding_components"],
                    "component_position": d["component_position"],
                    'n_preceding_comp_tokens': d['n_preceding_comp_tokens'],
                    'n_following_comp_tokens': d['n_following_comp_tokens'],
                    'first_in_paragraph': d['first_in_paragraph'],
                    'last_in_paragraph': d['last_in_paragraph']
                }
                # items.update(pd(d["cover_sentence"]))
                features.append(items)

            return features

        for d in feats:
            ret_tuple = (*ret_tuple, get_features(d))

        return ret_tuple

    def predict(self, data, probability=False) -> Union[list, bool]:
        if type(data) is str:
            data = [data]

        feats = self.__feats.transform(data)
        dict_feats = self.__dict_feats.transform(data)

        combined_feats = hstack([feats, dict_feats])
        combined_feats = self.__scaler.transform(combined_feats)
        return super().predict(combined_feats, probability)
