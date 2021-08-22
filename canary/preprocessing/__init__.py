from collections import Counter
from functools import cache

import nltk
import spacy
from nltk.corpus import wordnet

from canary.utils import nltk_download

_nlp = spacy.load("en_core_web_lg")
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

    def get_wordnet_pos(self, treebank_tag):

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

    @cache
    def __process(self, t):
        tag = nltk.pos_tag([t])[0][1]
        return _word_net.lemmatize(t, self.get_wordnet_pos(tag))

    def __call__(self, text):
        return [self.__process(t) for t in nltk.word_tokenize(text)]


class PosLemmatizer:

    def t(self, x):
        return f"{x.lemma_}/{x.tag_}"

    @cache
    def __call__(self, text):
        text = _nlp(text)
        return [self.t(d) for d in text]


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


class PosDistribution:

    def __init__(self):
        self.keys = {}

        keys = list(nltk.load('help/tagsets/upenn_tagset.pickle').keys())
        for key in keys:
            self.keys[key] = 0

    def __call__(self, text):
        tokens = nltk.word_tokenize(text)
        counts = Counter(tag for _, tag in nltk.pos_tag(tokens))
        count_dict = self.keys.copy()
        for tag, count in counts.items():
            count_dict[tag] = count

        return count_dict
