import pandas
from nltk.tree import Tree
from scipy.sparse import hstack
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import RobustScaler

import canary
import canary.utils
from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import DiscourseMatcher, FirstPersonIndicatorMatcher

_nlp = canary.preprocessing.nlp.spacy_download()


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

        # If the pipeline model is none, use this algorithm
        if pipeline_model is None:
            pipeline_model = make_pipeline(
                ArgumentComponentFeatures(),
                RobustScaler(with_centering=False, unit_variance=False),
                LogisticRegression(
                    class_weight='balanced',
                    warm_start=True,
                    random_state=0,
                    max_iter=2000,
                    solver="newton-cg"
                ))

        super(ArgumentComponent, self).train(
            pipeline_model=pipeline_model,
            train_data=train_data,
            test_data=test_data,
            train_targets=train_targets,
            test_targets=test_targets,
            save_on_finish=save_on_finish
        )


class ArgumentComponentFeatures(TransformerMixin, BaseEstimator):
    features: list = [
        TfidfVectorizer(ngram_range=(1, 3), tokenizer=Lemmatizer(), lowercase=False, max_features=1000),
        DiscourseMatcher('forward', lemmatize=True),
        DiscourseMatcher('thesis', lemmatize=True),
        DiscourseMatcher('rebuttal', lemmatize=True),
        DiscourseMatcher('backward', lemmatize=True),
        FirstPersonIndicatorMatcher("I"),
        FirstPersonIndicatorMatcher("me"),
        FirstPersonIndicatorMatcher("mine"),
        FirstPersonIndicatorMatcher("myself"),
        FirstPersonIndicatorMatcher("my"),
    ]

    def __init__(self):
        self.__dict_feats = FeatureHasher()
        self.__features = make_union(*ArgumentComponentFeatures.features)

    @staticmethod
    def prepare_dictionary_features(data):
        def get_features(feats):
            canary.utils.logger.debug("getting dictionary features.")
            features = []

            for d in feats:
                sen = _nlp(d['cover_sentence'])
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
                features.append(items)

            return features

        return get_features(data)

    def fit(self, x, y=None):
        self.__dict_feats.fit(x)
        self.__features.fit(pandas.DataFrame(x).cover_sentence.tolist())
        return self

    def transform(self, x):
        features = self.__features.transform(pandas.DataFrame(x).cover_sentence.tolist())
        dict_features = self.__dict_feats.transform(self.prepare_dictionary_features(x))

        return hstack([features, dict_features])
