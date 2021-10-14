"""Transformers for various NLP tasks. Emulates a scikit-learn like API."""

import string
from abc import ABCMeta
from collections import Counter
from functools import lru_cache

import nltk
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .._data.indicators import discourse_indicators
from ..nlp import Lemmatiser
from ..nlp import PunctuationTokeniser
from ..nlp._utils import spacy_download
from ..utils import logger

__all__ = [
    "SentimentTransformer",
    "WordSentimentCounter",
    "TfidfPosVectorizer",
    "CountPosVectorizer",
    "LengthOfSentenceTransformer",
    "UniqueWordsTransformer",
    "LengthTransformer",
    "AverageWordLengthTransformer",
    "DiscourseMatcher",
    "FirstPersonIndicatorMatcher",
    "CountPunctuationVectorizer",
    "TfidfPunctuationVectorizer",
    "EmbeddingTransformer",
    "BiasTransformer",
    "SharedNouns",
    "ProductionRules"
]


class PosVectorizer(metaclass=ABCMeta):
    """
    Base class for POS tagging vectorisation
    """

    nlp = spacy_download()

    def __init__(self, ngrams=1) -> None:

        if type(ngrams) is int:
            self.ngrams = ngrams

        super().__init__()

    def prepare_doc(self, doc):

        _doc = self.nlp(doc)
        new_text = []

        for word in _doc:
            new_text.append(f"{word.lemma_}/{word.tag_}")
        if self.ngrams != 1:
            new_text = list(nltk.ngrams(new_text, self.ngrams))

        return new_text


class SentimentTransformer(TransformerMixin, BaseEstimator):
    """
    Gets sentiment from a segment of text.
    """

    _analyser = SentimentIntensityAnalyzer()
    _allowed_targets = ["compound", "pos", "neg", "neu"]
    target = "compound"

    def __init__(self, target="compound") -> None:
        if target not in self._allowed_targets:
            logger.warn(
                f"{target} is not in the allowed value list: {self._allowed_targets}. Defaulting to 'compound'")
        else:
            self.target = target
        super().__init__()

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[self._analyser.polarity_scores(y)[self.target]] for y in x]


class WordSentimentCounter(TransformerMixin, BaseEstimator):
    """
    Counts the occurrences of sentiment in a text
    """

    _analyser = SentimentIntensityAnalyzer()
    _allowed_targets = ['pos', 'neg', 'neu']
    target = "pos"

    def __init__(self, target: str = "pos"):
        if target not in self._allowed_targets:
            logger.warn("")
        else:
            self.target = target

    def countSentiment(self, doc: str):
        occurrences = 0
        for token in doc.split():
            if self._analyser.polarity_scores(token)[self.target] >= 1.0:
                occurrences += 1
        return occurrences

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[self.countSentiment(y)] for y in x]


class TfidfPosVectorizer(PosVectorizer, TfidfVectorizer):
    """

    """

    def build_analyzer(self):
        def analyzer(doc):
            return self.prepare_doc(doc)

        return analyzer


class CountPosVectorizer(PosVectorizer, CountVectorizer):
    """

    """

    def build_analyzer(self):
        def analyzer(doc):
            return self.prepare_doc(doc)

        return analyzer


class LengthOfSentenceTransformer(TransformerMixin, BaseEstimator):
    """
    Returns the length of a sentence
    """

    def fit(self, x, y):
        return self

    def num_of_words(self, x):
        return len(x.split)

    def transform(self, x):
        return [[len(y.split())] for y in x]


class UniqueWordsTransformer(TransformerMixin, BaseEstimator):
    """
     Returns the number of unique words in a sentence
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[len(np.unique(nltk.word_tokenize(y)))] for y in x]


class LengthTransformer(TransformerMixin, BaseEstimator):
    """
    Determins if sentence is longer than a certain length
    """

    def __init__(self, word_length: int = None):
        if word_length is None:
            self.word_length = 5
        else:
            self.word_length = word_length

        super(LengthTransformer, self).__init__()

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[len(y.split()) > self.word_length] for y in x]


class AverageWordLengthTransformer(TransformerMixin, BaseEstimator):
    """
    Calculates the average word length for a document
    """

    def fit(self, x, y):
        return self

    @staticmethod
    def average(x):
        words = x.split()
        av = sum(map(len, words)) / len(words)
        return av

    def transform(self, x):
        return [[self.average(y)] for y in x]


class DiscourseMatcher(TransformerMixin, BaseEstimator):
    """
    Checks for indications of discourse in text

    indicators can be seen in data/indicators.py
    """

    def __init__(self, component=None, lemmatise=False):
        self.component = None
        self.lemmatise = lemmatise
        self._test = 1

        if component is not None and component not in discourse_indicators.keys():
            raise ValueError(
                f"Incorrect discourse component passed to constructor. "
                f"Acceptable values are {[k for k in discourse_indicators.keys()]}")

    @property
    def indicators(self) -> list:
        indicators = []
        if self.component is None:
            for x in discourse_indicators.keys():
                for y in discourse_indicators[x]:
                    indicators.append(y)
        else:
            indicators = discourse_indicators[self.component]

        return indicators

    def fit(self, x, y):
        return self

    @lru_cache(maxsize=None)
    def __contains_indicator__(self, sen) -> bool:
        for x in self.indicators:
            if type(sen) is str:
                sen = sen.split()
            if x in sen:
                return True
            if hasattr(self, 'lemmatise'):
                if self.lemmatise is True:
                    sen = [k.lower() for k in sen]
                    lemma = nltk.WordPunctTokenizer().tokenize(x)[0].lower()
                    if lemma.lower() in sen:
                        return True
        return False

    def transform(self, doc):
        return [[self.__contains_indicator__(x)] for x in doc]


class FirstPersonIndicatorMatcher(TransformerMixin, BaseEstimator):
    """Matches if any first-person indicators are present in text"""

    def __init__(self, indicator=None, lemmatise=False):
        self.indicator = indicator
        self.lemmatise = False

    @property
    def indicators(self):
        return discourse_indicators['first_person']

    def fit(self, x, y):
        return self

    def __contains_indicator__(self, sen):

        sen = [k.lower() for k in sen.split()]
        for x in self.indicators:
            if x.lower() in sen:
                return True
            if self.lemmatise is True:
                sen = [k.lower() for k in sen]
                lemma = nltk.WordPunctTokenizer().tokenize(x)[0].lower()
                if lemma.lower() in sen:
                    return True

        return False

    def transform(self, doc):
        return [[self.__contains_indicator__(x)] for x in doc]


class CountPunctuationVectorizer(CountVectorizer):
    """

    """

    def __init__(self):
        self.punctuation = [character for character in (string.punctuation + "£")]
        super().__init__(tokenizer=PunctuationTokeniser())

    def prepare_doc(self, doc):
        _doc = doc
        _doc = _doc.replace("\\r\\n", " ")
        for character in (_doc):
            if character not in self.punctuation:
                _doc = _doc.replace(character, "", 1)
        return _doc

    def build_analyzer(self):
        def analyzer(doc):
            p = self.build_preprocessor()
            return p(self.prepare_doc(doc))

        return analyzer


class TfidfPunctuationVectorizer(TfidfVectorizer):
    """

    """

    def __init__(self):
        self.punctuation = [character for character in string.punctuation]
        super().__init__(tokenizer=PunctuationTokeniser())

    def prepare_doc(self, doc):
        _doc = doc
        _doc = _doc.replace("\\r\\n", " ")
        for character in (_doc):
            if character not in self.punctuation:
                _doc = _doc.replace(character, "", 1)
        return _doc

    def build_analyzer(self):
        def analyzer(doc):
            p = self.build_preprocessor()
            return p(self.prepare_doc(doc))

        return analyzer


class EmbeddingTransformer(TransformerMixin, BaseEstimator):
    _nlp = spacy_download(
        disable=['ner', 'textcat', 'tagger', 'lemmatizer', 'tokenizer',
                 'attribute_ruler',
                 'benepar'])

    def fit(self, x, y):
        return self

    def transform(self, x):
        x = self._nlp.pipe(x)
        return [[y.vector_norm] for y in x]


class BiasTransformer(TransformerMixin, BaseEstimator):
    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[True] for _ in x]


class SharedNouns(TransformerMixin, BaseEstimator):
    lemmatiser = Lemmatiser()

    def fit(self, x, y=None):
        return self

    def transform(self, arg1, arg2):
        nouns_in_arg1 = [self.lemmatiser(word)[0] for (word, pos) in nltk.pos_tag(nltk.word_tokenize(arg1)) if
                         (pos[:2] == 'NN')]
        nouns_in_arg2 = [self.lemmatiser(word)[0] for (word, pos) in nltk.pos_tag(nltk.word_tokenize(arg2)) if
                         (pos[:2] == 'NN')]

        return len(set(nouns_in_arg1).intersection(nouns_in_arg2))


class ProductionRules(TransformerMixin, BaseEstimator):

    def __init__(self, max_rules=500):
        self._production_rules = []
        self._n_rules = max_rules
        self.__nlp = spacy_download(
            disable=['ner', 'textcat', 'tagger', 'lemmatizer', 'tokenizer',
                     'attribute_ruler',
                     'tok2vec'])

    @property
    def production_rules(self):
        return self._production_rules

    def fit(self, x, y=None):
        logger.debug(f"Fitting {self.__class__.__name__}")
        self._production_rules = []

        x = list(self.__nlp.pipe(x))

        for item in x:

            productions = self.get_productions(item)
            for p in productions:
                self._production_rules.append(p)

        self._production_rules = Counter(self.production_rules).most_common(self._n_rules)
        self._production_rules = [str(p[0]) for p in self.production_rules]

        return self

    @staticmethod
    @lru_cache(maxsize=None)
    def get_productions(item):
        parse_string = list(item.sents)[0]._.parse_string
        tree = nltk.Tree.fromstring(parse_string)
        return tree.productions()

    @lru_cache(maxsize=None)
    def process_item(self, x):
        productions = self.get_productions(x)
        productions = [str(p) for p in productions]

        out = [0 for _ in self.production_rules]
        for i, rule in enumerate(self.production_rules):
            if rule in productions:
                out[i] = 1
        return out

    def transform(self, x):
        x = self.__nlp.pipe(x)
        return [self.process_item(_x) for _x in x]
