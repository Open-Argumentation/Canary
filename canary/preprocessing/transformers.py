import string

import nltk
import numpy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from canary.data.indicators import discourse_indicators
from canary.preprocessing import PunctuationTokenizer


class SentimentTransformer(TransformerMixin, BaseEstimator):
    """
        gets sentiment from a segment of text.
    """
    _analyser = SentimentIntensityAnalyzer()

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[self._analyser.polarity_scores(y)['compound']] for y in x]


class TfidfPosVectorizer(TfidfVectorizer):
    """

    """

    def prepare_doc(self, doc):
        _doc = nltk.WordPunctTokenizer().tokenize(doc)
        _doc = [nltk.WordNetLemmatizer().lemmatize(token) for token in _doc]
        _doc = nltk.pos_tag(_doc)
        new_text = []
        for word in _doc:
            new_text.append(word[1])
        return new_text

    def build_analyzer(self):
        def analyzer(doc):
            return self.prepare_doc(doc)

        return analyzer


class CountPosVectorizer(CountVectorizer):
    """

    """

    def prepare_doc(self, doc):
        _doc = nltk.WordPunctTokenizer().tokenize(doc)
        _doc = [nltk.WordNetLemmatizer().lemmatize(token) for token in _doc]
        _doc = nltk.pos_tag(_doc)
        new_text = []
        for word in _doc:
            new_text.append(word[1])
        return new_text

    def build_analyzer(self):
        def analyzer(doc):
            return self.prepare_doc(doc)

        return analyzer


class LengthOfSentenceTransformer(TransformerMixin, BaseEstimator):
    """

    """

    def fit(self, x, y):
        return self

    def num_of_words(self, x):
        return len(x.split)

    def transform(self, x):
        return [[len(y.split())] for y in x]


class UniqueWordsTransformer(TransformerMixin, BaseEstimator):
    """

    """

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[len(numpy.unique(nltk.word_tokenize(y)))] for y in x]


class LengthTransformer(TransformerMixin, BaseEstimator):
    """
    Returns
    """

    def __init__(self, word_length=None):
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

    def average(self, x):
        words = x.split()
        av = sum(map(len, words)) / len(words)
        return av

    def transform(self, x):
        return [[self.average(y)] for y in x]


class DiscourseMatcher(TransformerMixin, BaseEstimator):
    """

    """
    component = None

    def __init__(self, component=None):
        if component is not None and component not in discourse_indicators.keys():
            raise ValueError(
                f"Incorrect discourse component passed to constructor. Acceptable values are {[k for k in discourse_indicators.keys()]}")
        else:
            self.component = component

    @property
    def indicators(self):
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

    def __contains_indicator__(self, sen):
        for x in self.indicators:
            if x in sen:
                return True
        return False

    def transform(self, doc):
        return [[self.__contains_indicator__(x)] for x in doc]


class FirstPersonIndicatorMatcher(TransformerMixin, BaseEstimator):
    """
    """

    def __init__(self, indicator):
        self.indicator = indicator

    @property
    def indicators(self):
        return discourse_indicators['first_person']

    def fit(self, x, y):
        return self

    def __contains_indicator__(self, sen):
        for x in self.indicators:
            if x in sen:
                return True
        return False

    def transform(self, doc):
        return [[self.__contains_indicator__(x)] for x in doc]


class CountPunctuationVectorizer(CountVectorizer):

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


class TfidfPunctuationVectorizer(TfidfVectorizer):

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
