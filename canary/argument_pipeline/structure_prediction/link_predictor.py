import nltk
import pandas
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import LabelBinarizer, MaxAbsScaler
from sklearn.svm import SVC

from canary.argument_pipeline.base import Model
from canary.preprocessing import PosDistribution
from canary.preprocessing.nlp import spacy_download, nltk_download
from canary.preprocessing.transformers import DiscourseMatcher, SharedNouns
from canary.utils import logger

nltk_download('punkt')
_nlp = spacy_download(disable=['ner', 'textcat', 'tagger', 'lemmatizer', 'tokenizer',
                               'attribute_ruler',
                               'benepar'])

__all__ = [
    "LinkPredictor",
    "LinkFeatures"
]


class LinkPredictor(Model):
    """
    Prediction model which can predict if two argument components are "linked".
    """

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            model_id = "link_predictor"

        super().__init__(model_id)

    @staticmethod
    def default_train():
        """
        Default training method
        :return: training data, test data, training targets, test targets
        """
        from canary.corpora import load_essay_corpus
        from imblearn.over_sampling import RandomOverSampler
        from sklearn.model_selection import train_test_split

        ros = RandomOverSampler(random_state=0, sampling_strategy=0.5)
        x, y = load_essay_corpus(purpose='link_prediction')
        x, y = ros.fit_resample(pandas.DataFrame(x), pandas.DataFrame(y))

        train_data, test_data, train_targets, test_targets = \
            train_test_split(x, y,
                             train_size=0.5,
                             shuffle=True,
                             random_state=0,
                             )

        logger.debug("Resample")

        return list(train_data.to_dict("index").values()), list(test_data.to_dict("index").values()), train_targets[
            0].tolist(), test_targets[0].tolist()

    @classmethod
    def train(cls, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):

        if pipeline_model is None:
            pipeline_model = make_pipeline(
                LinkFeatures(),
                MaxAbsScaler(),
                SVC(random_state=0, probability=True, C=10),
            )

        return super().train(pipeline_model, train_data, test_data, train_targets, test_targets, save_on_finish,
                             *args,
                             **kwargs)


class LinkFeatures(TransformerMixin, BaseEstimator):
    """
    Transformer which handles LinkPredictor features
    """

    feats: list = [
        DiscourseMatcher('forward'),
        DiscourseMatcher('thesis'),
        DiscourseMatcher('rebuttal'),
        DiscourseMatcher('backward'),
    ]

    def __init__(self):

        self.__nom_dict_features = DictVectorizer()

        self.__numeric_dict_features = DictVectorizer()

        self.__arg1_cover_features = make_union(*LinkFeatures.feats)

        self.__arg2_cover_features = make_union(*LinkFeatures.feats)

        self.__ohe_arg1 = LabelBinarizer()

        self.__ohe_arg2 = LabelBinarizer()

    def fit(self, x, y=None):
        """
        fits self to data

        :param x:
        :param y: ignored.
        :return:
        """
        px = pandas.DataFrame(x)

        self.__arg1_cover_features.fit(px.arg1_covering_sentence.tolist())
        self.__arg2_cover_features.fit(px.arg2_covering_sentence.tolist())

        self.__numeric_dict_features.fit(self.prepare_numeric_feats(x))
        self.__nom_dict_features.fit(self.prepare_dictionary_features(x))

        self.__ohe_arg1.fit(["Premise", "Claim", "MajorClaim"])
        self.__ohe_arg2.fit(["Premise", "Claim", "MajorClaim"])

        return self

    def transform(self, x):
        dict_feats = self.__nom_dict_features.transform(self.prepare_dictionary_features(x))
        num_dict_feats = self.__numeric_dict_features.transform(self.prepare_numeric_feats(x))

        x = pandas.DataFrame(x)

        arg1_cover_feats = self.__arg1_cover_features.transform(x.arg1_covering_sentence)
        arg2_cover_feats = self.__arg2_cover_features.transform(x.arg2_covering_sentence)

        arg1_types = self.__ohe_arg1.transform(x.arg1_type)
        arg2_types = self.__ohe_arg2.transform(x.arg2_type)

        return hstack([dict_feats, num_dict_feats, arg1_cover_feats, arg2_cover_feats, arg1_types, arg2_types])

    @staticmethod
    def prepare_dictionary_features(data):
        logger.debug("Getting dictionary features")
        shared_noun_counter = SharedNouns()

        def get_features(feats):
            new_feats = feats.copy()

            for t, f in enumerate(new_feats):
                n_shared_nouns = shared_noun_counter.transform(f["arg1_component"], f["arg2_component"])

                features = {
                    "source_before_target": f.get("source_before_target"),
                    "arg1_first_in_paragraph": f.get("arg1_first_in_paragraph"),
                    "arg1_last_in_paragraph": f["arg1_last_in_paragraph"],
                    "arg2_first_in_paragraph": f["arg2_first_in_paragraph"],
                    "arg2_last_in_paragraph": f["arg2_last_in_paragraph"],
                    "arg1_is_premise": f["arg1_type"] == "Premise",
                    "arg1_in_intro": f["arg1_in_intro"],
                    "arg1_in_conclusion": f["arg1_in_conclusion"],
                    "arg2_in_intro": f['arg2_in_intro'],
                    "arg2_in_conclusion": f["arg2_in_conclusion"],
                    "arg1_and_arg2_in_same_sentence": f["arg1_and_arg2_in_same_sentence"],
                    "arg1_indicator_type_follows_component": f["arg1_indicator_type_follows_component"],
                    "arg2_indicator_type_follows_component": f["arg2_indicator_type_follows_component"],
                    "arg1_indicator_type_precedes_component": f["arg1_indicator_type_precedes_component"],
                    "arg2_indicator_type_precedes_component": f["arg2_indicator_type_precedes_component"],
                    "share_nouns": n_shared_nouns > 0,
                }

                new_feats[t] = features
            return new_feats

        return get_features(data)

    @staticmethod
    def prepare_numeric_feats(data):
        shared_noun_counter = SharedNouns()
        logger.debug("Getting numeric features")

        def get_features(feats):
            pos_dist = PosDistribution()
            arg1_covering_sentence = pandas.DataFrame(feats).arg1_covering_sentence.tolist()
            arg1_covering_sentence = list(_nlp.pipe(arg1_covering_sentence))

            arg2_covering_sentence = pandas.DataFrame(feats).arg2_covering_sentence.tolist()
            arg2_covering_sentence = list(_nlp.pipe(arg2_covering_sentence))

            new_feats = feats.copy()
            for t, f in enumerate(new_feats):
                n_shared_nouns = shared_noun_counter.transform(f["arg1_component"], f["arg2_component"])

                features = {
                    "n_para_components": f['n_para_components'],
                    "n_components_between_pair": abs(f["arg2_position"] - f["arg1_position"]),
                    "arg1_component_token_len": len(nltk.word_tokenize(f['arg1_component'])),
                    "arg2_component_token_len": len(nltk.word_tokenize(f['arg2_component'])),
                    "arg1_cover_sen_token_len": len(nltk.word_tokenize(f['arg1_covering_sentence'])),
                    "arg2_cover_sen_token_len": len(nltk.word_tokenize(f['arg2_covering_sentence'])),
                    "shared_nouns": n_shared_nouns,
                    "similarity": arg1_covering_sentence[t].similarity(arg2_covering_sentence[t])
                }

                arg1_posd = {"arg1_" + str(key): val for key, val in pos_dist(f['arg1_covering_sentence']).items()}
                arg2_posd = {"arg2_" + str(key): val for key, val in pos_dist(f['arg2_covering_sentence']).items()}

                features.update(arg1_posd)
                features.update(arg2_posd)

                new_feats[t] = features
            return new_feats

        return get_features(data)
