from collections import Counter

import nltk
from nltk.corpus import wordnet

import canary.preprocessing.nlp
from canary.preprocessing.nlp import nltk_download

_nlp = None
_word_net = nltk.WordNetLemmatizer()
_stemmer = nltk.PorterStemmer()
nltk_download(['punkt', 'wordnet', 'tagsets'])


class Lemmatizer:
    """
    Transforms text into its lemma form

    e.g.
    - cats -> cat
    - corpora -> corpus
    """

    @staticmethod
    def get_wordnet_pos(treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def __process(self, t):
        tag = nltk.pos_tag([t])[0][1]
        return _word_net.lemmatize(t, self.get_wordnet_pos(tag))

    def __call__(self, text):
        return [self.__process(t) for t in nltk.word_tokenize(text)]


class PosLemmatizer:

    def t(self, x):
        tag = nltk.pos_tag([x])[0][1]

        return f"{_word_net.lemmatize(x, Lemmatizer.get_wordnet_pos(tag))}/{tag}"

    def __call__(self, text):
        return [self.t(d) for d in nltk.word_tokenize(text)]


class Stemmer:
    """
    Transforms text into its stemmed form
    """

    def __call__(self, text):
        return [_stemmer.stem(token) for token in nltk.word_tokenize(text)]


class PunctuationTokenizer:
    """
    Extracts only punctuation from a piece of text

    e.g.
    - Hi, what's up? Did you like the movie last night?
    -> [[','], ["'", 's'], ['?'], ['?']]
    """

    def __init__(self):
        self.__tokenizer = nltk.WordPunctTokenizer()

    def __call__(self, text):
        return [self.__tokenizer.tokenize(t) for t in nltk.word_tokenize(text) if not t.isalnum()]


keys = list(nltk.load('help/tagsets/upenn_tagset.pickle').keys())


class PosDistribution:

    def __init__(self):
        self.keys = {}

        for key in keys:
            self.keys[key] = 0

    def __call__(self, text):
        tokens = nltk.word_tokenize(text)
        counts = Counter(tag for _, tag in nltk.pos_tag(tokens))
        count_dict = self.keys.copy()
        for tag, count in counts.items():
            count_dict[tag] = count

        return count_dict


class Tokenizer:

    def __init__(self):
        pass
