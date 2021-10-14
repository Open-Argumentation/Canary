import pandas
from nltk.tree import Tree
from scipy.sparse import hstack
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

from ..argument_pipeline.base import Model
from ..corpora import load_essay_corpus
from ..nlp import Lemmatiser, PosDistribution
from ..nlp.transformers import DiscourseMatcher, EmbeddingTransformer
from ..nlp._utils import spacy_download
from ..utils import logger

_nlp = spacy_download(disable=['ner', 'textcat', 'tagger', 'lemmatizer', 'tokenizer',
                               'attribute_ruler',
                               'tok2vec', ])

__all__ = [
    "ArgumentComponent",
    "ArgumentComponentFeatures"
]


class ArgumentComponent(Model):
    """Detects argumentative components from natural language e.g. premises and claims"""

    def __init__(self, model_id: str = None):

        if model_id is None:
            model_id = "argument_component"

        super().__init__(
            model_id=model_id,
        )

    @staticmethod
    def default_train():
        """The default training method. ArgumentComponent defaults to using the essay corpus with undersampling."""
        from sklearn.model_selection import train_test_split
        from imblearn.under_sampling import RandomUnderSampler
        ros = RandomUnderSampler(random_state=0, sampling_strategy='not minority')

        x, y = load_essay_corpus(purpose="component_prediction")
        x, y = ros.fit_resample(pandas.DataFrame(x), pandas.DataFrame(y))

        train_data, test_data, train_targets, test_targets = \
            train_test_split(x, y,
                             train_size=0.7,
                             shuffle=True,
                             random_state=0,
                             stratify=y
                             )

        logger.debug("Resample")

        return list(train_data.to_dict("index").values()), list(test_data.to_dict("index").values()), train_targets[
            0].tolist(), test_targets[0].tolist()

    @classmethod
    def train(cls, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):

        # If the pipeline model is none, use this algorithm
        if pipeline_model is None:
            pipeline_model = make_pipeline(
                ArgumentComponentFeatures(),
                MaxAbsScaler(),
                SVC(random_state=0, class_weight='balanced', probability=True, cache_size=1000)
            )

        return super().train(
            pipeline_model=pipeline_model,
            train_data=train_data,
            test_data=test_data,
            train_targets=train_targets,
            test_targets=test_targets,
            save_on_finish=save_on_finish
        )


class ArgumentComponentFeatures(TransformerMixin, BaseEstimator):
    """Transformer Mixin that extracts features for the ArgumentComponent model"""

    features: list = [
        TfidfVectorizer(ngram_range=(1, 1), tokenizer=Lemmatiser(), lowercase=False, binary=True),
        TfidfVectorizer(ngram_range=(2, 2), tokenizer=Lemmatiser(), lowercase=False, max_features=2000),
        DiscourseMatcher('forward'),
        DiscourseMatcher('thesis'),
        DiscourseMatcher('rebuttal'),
        DiscourseMatcher('backward'),
        DiscourseMatcher('obligation'),
        DiscourseMatcher('recommendation'),
        DiscourseMatcher('possible'),
        DiscourseMatcher('intention'),
        DiscourseMatcher('option'),
        DiscourseMatcher('first_person'),
        EmbeddingTransformer()
    ]

    def __init__(self):
        self.__dict_feats = DictVectorizer()
        self.__features = make_union(*ArgumentComponentFeatures.features)

    @staticmethod
    def _prepare_dictionary_features(data):
        pos_dist = PosDistribution()
        cover_sentences = pandas.DataFrame(data).cover_sentence.tolist()
        cover_sentences = list(_nlp.pipe(cover_sentences))

        def get_features(feats):
            features = []

            for i, d in enumerate(feats):
                cover_sen_parse_tree = Tree.fromstring(list(cover_sentences[i].sents)[0]._.parse_string)

                items = {
                    'tree_height': cover_sen_parse_tree.height(),
                    'len_paragraph': d.get('len_paragraph'),
                    "len_component": d.get('len_component'),
                    "len_cover_sen": d.get('len_cover_sen'),
                    'is_in_intro': d.get('is_in_intro'),
                    'is_in_conclusion': d.get('is_in_conclusion'),
                    "n_following_components": d.get("n_following_components"),
                    "n_preceding_components": d.get("n_preceding_components"),
                    "component_position": d.get("component_position"),
                    'n_preceding_comp_tokens': d.get('n_preceding_comp_tokens'),
                    'n_following_comp_tokens': d.get('n_following_comp_tokens'),
                    'first_in_paragraph': d.get('first_in_paragraph'),
                    'last_in_paragraph': d.get('last_in_paragraph')
                }
                items.update(pos_dist(d['cover_sentence']).items())
                features.append(items)

            return features

        return get_features(data)

    def fit(self, x: list, y: list = None):
        """Fits self to data provided.

        Parameters
        ----------
        x: list
            The data on which the transformer is fitted.
        y: list
            Ignored. Providing will have no effect. Provided for compatibility reasons.

        Returns
        -------
        Self
        """
        logger.debug("Fitting")
        self.__dict_feats.fit(x)
        self.__features.fit(pandas.DataFrame(x).cover_sentence.tolist())
        return self

    def transform(self, x: list):
        """Transforms data provided.

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
        features = self.__features.transform(pandas.DataFrame(x).cover_sentence.tolist())
        dict_features = self.__dict_feats.transform(self._prepare_dictionary_features(x))

        return hstack([features, dict_features])
