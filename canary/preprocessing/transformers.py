import string
from abc import ABCMeta

import nltk
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from canary import logger
from canary.data.indicators import discourse_indicators
from canary.preprocessing import PunctuationTokenizer


# @TODO this file needs cleaning up

class PosVectorizer(metaclass=ABCMeta):
    """
    Base class for POS tagging vectorisation
    """

    bigrams = False

    def __init__(self, bigrams=False) -> None:
        if bigrams is not False:
            self.bigrams = bigrams

        super().__init__()

    def prepare_doc(self, doc):
        _doc = nltk.WordPunctTokenizer().tokenize(doc)

        # @TODO improve lemmatising. Only considers nouns.
        _doc = [nltk.WordNetLemmatizer().lemmatize(token) for token in _doc]
        _doc = nltk.pos_tag(_doc)
        new_text = []

        for word in _doc:
            new_text.append(word[1])
        if self.bigrams is True:
            new_text = list(nltk.bigrams(new_text))
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


class TfidfPosVectorizer(TfidfVectorizer, PosVectorizer):
    """

    """

    def build_analyzer(self):
        def analyzer(doc):
            return self.prepare_doc(doc)

        return analyzer


class CountPosVectorizer(CountVectorizer, PosVectorizer):
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

    component = None
    lemmatize = False

    def __init__(self, component=None, lemmatize=False):
        if component is not None and component not in discourse_indicators.keys():
            raise ValueError(
                f"Incorrect discourse component passed to constructor. "
                f"Acceptable values are {[k for k in discourse_indicators.keys()]}")
        else:
            self.component = component

        if lemmatize is not False and type(lemmatize) is bool:
            self.lemmatize = lemmatize

    @property
    def indicators(self) -> list[str]:
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

    def __contains_indicator__(self, sen) -> bool:
        for x in self.indicators:
            if type(sen) is str:
                sen = sen.split()
            if x in sen:
                return True
            sen = [k.lower() for k in sen]
            lemma = nltk.WordPunctTokenizer().tokenize(x)[0].lower()
            if lemma.lower() in sen:
                return True
        return False

    def transform(self, doc):
        return [[self.__contains_indicator__(x)] for x in doc]


class FirstPersonIndicatorMatcher(TransformerMixin, BaseEstimator):
    """
    Matches if any first-person indicators are present in text
    """

    def __init__(self, indicator):
        self.indicator = indicator

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
        super().__init__(tokenizer=PunctuationTokenizer())

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
        super().__init__(tokenizer=PunctuationTokenizer())

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
