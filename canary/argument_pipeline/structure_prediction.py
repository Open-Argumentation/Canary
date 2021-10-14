"""The structure prediction module provides functionality in respect toe the prediction
and structure of a document.
"""
import pandas
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import LabelBinarizer, Normalizer

from ..argument_pipeline.base import Model
from ..nlp._utils import spacy_download
from ..nlp.transformers import WordSentimentCounter, DiscourseMatcher
from ..utils import logger

nlp = spacy_download()

__all__ = [
    "StructurePredictor",
    "StructureFeatures"
]


class StructurePredictor(Model):

    def __init__(self, model_id=None):
        if model_id is None:
            model_id = "structure_predictor"

        super().__init__(model_id=model_id)

    @staticmethod
    def default_train():
        """Default training method which supplies the default training set"""

        from canary.corpora import load_essay_corpus
        from imblearn.over_sampling import RandomOverSampler
        from sklearn.model_selection import train_test_split

        ros = RandomOverSampler(random_state=0, sampling_strategy=0.5)
        x, y = load_essay_corpus(purpose="relation_prediction")
        x, y = ros.fit_resample(pandas.DataFrame(x), pandas.DataFrame(y))

        train_data, test_data, train_targets, test_targets = \
            train_test_split(x, y,
                             train_size=0.6,
                             shuffle=True,
                             random_state=0,
                             )

        logger.debug("Resample")

        return list(train_data.to_dict("index").values()), list(test_data.to_dict("index").values()), train_targets[
            0].tolist(), test_targets[0].tolist()

    @classmethod
    def train(cls, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, **kwargs):

        if pipeline_model is None:
            pipeline_model = make_pipeline(
                StructureFeatures(),
                Normalizer(),
                SGDClassifier(random_state=0, loss="log", ))

        return super().train(
            pipeline_model=pipeline_model,
            train_data=train_data,
            test_data=test_data,
            train_targets=train_targets,
            test_targets=test_targets,
            save_on_finish=save_on_finish
        )


class StructureFeatures(TransformerMixin, BaseEstimator):
    """A custom feature transformer used for extracting features relevant to structure prediction"""
    cover_features: list = [
        DiscourseMatcher("support"),
        DiscourseMatcher("conflict"),
        DiscourseMatcher('forward'),
        DiscourseMatcher('thesis'),
        WordSentimentCounter("neu"),
        WordSentimentCounter("pos"),
        WordSentimentCounter("neg")
    ]

    def __init__(self):
        self.__arg1_cover_features = make_union(*StructureFeatures.cover_features)
        self.__arg2_cover_features = make_union(*StructureFeatures.cover_features)

        self.__dictionary_features = DictVectorizer()

        self.__ohe_arg1 = LabelBinarizer()
        self.__ohe_arg2 = LabelBinarizer()

    def fit(self, x, y=None):
        """Fits self to data provided.

        Parameters
        ----------
        x: list
            The data on which the transformer is fitted.
        y: list, default=None
            Ignored. Providing will have no effect. Provided for compatibility reasons.

        Returns
        -------
        self
        """

        self.__dictionary_features.fit(self._prepare_dictionary_features(x))
        x = pandas.DataFrame(x)

        self.__arg1_cover_features.fit(x.arg1_covering_sentence.tolist())
        self.__arg2_cover_features.fit(x.arg2_covering_sentence.tolist())

        self.__ohe_arg1.fit(x.arg1_type.tolist())
        self.__ohe_arg2.fit(x.arg2_type.tolist())
        return self

    def transform(self, x) -> list:
        """Transform data into features

        Parameters
        ----------
        x: list
            A list of datapoints which are to be transformed using the mixin

        Returns
        -------
        scipy.sparse.hstack
            The features of the inputted list

        See Also
        ---------
        scipy.sparse.hstack
        """
        dictionary_features = self.__dictionary_features.transform(x)

        x = pandas.DataFrame(x)
        arg1_cover_features = self.__arg1_cover_features.transform(x.arg1_covering_sentence)
        arg2_cover_features = self.__arg2_cover_features.transform(x.arg2_covering_sentence)

        arg1_types = self.__ohe_arg1.transform(x.arg1_type)
        arg2_types = self.__ohe_arg2.transform(x.arg2_type)

        return hstack(
            [
                dictionary_features,
                arg1_cover_features,
                arg2_cover_features,
                arg1_types,
                arg2_types
            ]
        )

    @staticmethod
    def _binary_neg_present(sen: str):
        """Detects if negative words are present

        Parameters
        ----------
        sen: str
            A sentence of natural language

        Returns
        -------
        bool
            A boolean indicating if negative words are present
        """
        return WordSentimentCounter("neg").transform([sen])[0][0] > 0

    @staticmethod
    def _prepare_dictionary_features(data: dict):
        """Takes a dictionary and extracts relevant features

        Parameters
        ----------
        data: a dictionary of relevant features

        Returns
        -------
        dict:
            The new dictionary
        """

        def get_features(f):
            new_feats = f.copy()
            for t, d in enumerate(new_feats):
                sent1 = nlp(d["arg1_covering_sentence"])
                sent2 = nlp(d["arg2_covering_sentence"])
                new_feats[t] = {
                    "arg1_position": d["arg1_position"],
                    "arg2_position": d["arg2_position"],
                    'arg1_preceding_tokens': d['arg1_preceding_tokens'],
                    "arg1_following_tokens": d["arg1_following_tokens"],
                    'arg2_preceding_tokens': d['arg2_preceding_tokens'],
                    "arg2_following_tokens": d["arg2_following_tokens"],
                    "sentence_similarity_norm": sent1.similarity(sent2),
                    "n_preceding_components": d["n_preceding_components"],
                    "n_following_components": d["n_following_components"],
                    "n_attack_components": d["n_attack_components"],
                    "n_support_components": d["n_support_components"],
                    "neg_present_arg1": StructureFeatures._binary_neg_present(sent1.text),
                    "neg_present_arg2": StructureFeatures._binary_neg_present(sent2.text),
                }
            return new_feats

        return get_features(data)
