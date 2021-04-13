import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from canary.data.indicators import discourse_indicators
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyser = SentimentIntensityAnalyzer()


class SentimentTransformer(object):
    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[_analyser.polarity_scores(y)['compound']] for y in x]


class TfidfPosVectorizer(TfidfVectorizer):

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


class LengthOfSentenceTransformer(object):

    def fit(self, x, y):
        return self

    def num_of_words(self, x):
        return len(x.split)

    def transform(self, x):
        return [[len(y.split())] for y in x]


class LengthTransformer(object):

    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[len(y.split()) > 12] for y in x]


class AverageWordLengthTransformer(object):

    def fit(self, x, y):
        return self

    def average(self, x):
        words = x.split()
        av = sum(map(len, words)) / len(words)
        return av

    def transform(self, x):
        return [[self.average(y)] for y in x]


class DiscourseMatcher(object):

    def __init__(self, component=None):
        self.component = component

    @property
    def indicators(self):
        if self.component is None:
            indicators = discourse_indicators['claim'] + discourse_indicators['major_claim'] + discourse_indicators[
                'premise']
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


class FirstPersonIndicatorMatcher(object):
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


class CountPunctuationVectorizer(TfidfVectorizer):
    _punctuation = discourse_indicators['punctuation']

    @property
    def punctuation(self):
        return self._punctuation

    @punctuation.setter
    def punctuation(self, value):
        self._punctuation = value

    def prepare_doc(self, doc):
        _doc = doc
        _doc = _doc.replace("\\r\\n", " ")
        for character in _doc:
            if character not in self.punctuation:
                _doc = _doc.replace(character, "")
        return _doc

    def build_analyzer(self):
        def analyzer(doc):
            p = self.build_preprocessor()
            return p(self.decode(self.prepare_doc(doc)))

        return analyzer


class TfidfPunctuationVectorizer(TfidfVectorizer):
    _punctuation = discourse_indicators['punctuation']

    @property
    def punctuation(self):
        return self._punctuation

    @punctuation.setter
    def punctuation(self, value):
        self._punctuation = value

    def prepare_doc(self, doc):
        _doc = doc
        _doc = _doc.replace("\\r\\n", " ")
        for character in _doc:
            if character not in self.punctuation:
                _doc = _doc.replace(character, "")
        return _doc

    def build_analyzer(self):
        def analyzer(doc):
            p = self.build_preprocessor()
            return p(self.decode(self.prepare_doc(doc)))

        return analyzer
