from collections import Counter
from functools import lru_cache

import nltk
from nltk.corpus import wordnet

import canary.nlp._utils
from canary.nlp._utils import nltk_download

_nlp = None
_word_net = nltk.WordNetLemmatizer()
_stemmer = nltk.PorterStemmer()
nltk_download(['punkt', 'wordnet', 'tagsets'])

__all__ = [
    "Lemmatiser",
    "PunctuationTokeniser",
    "PosDistribution",
]


class Lemmatiser:
    """Transforms text into its lemma form

    Notes
    -----
        e.g.
        - cats -> cat
        - corpora -> corpus
    """

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """Internal helper to get pos tag type

        The pos tag type is needed for the lemmatize function which is more accurate if the pos tag is known.

        Parameters
        ----------
        treebank_tag: str
            some....

        Returns
        -------
        str
            The pos tag type

        See Also
        --------
        nltk.tag.pos_tag, nltk.stem.wordnet
        """
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

    @lru_cache(maxsize=None)
    def __process(self, t):
        tag = nltk.pos_tag([t])[0][1]
        return _word_net.lemmatize(t, self.get_wordnet_pos(tag))

    def __call__(self, text):
        return [self.__process(t) for t in nltk.word_tokenize(text)]


class PunctuationTokeniser:
    """Extracts only punctuation from a piece of text

    Notes
    -----
        e.g.
        - Hi, what's up? Did you like the movie last night?
        -> [[','], ["'", 's'], ['?'], ['?']]

    See Also
    --------
    nltk.tokenize.regexp.WordPunctTokenize
    """

    def __init__(self):
        self.__tokenizer = nltk.WordPunctTokenizer()

    def __call__(self, text):
        return [self.__tokenizer.tokenize(t) for t in nltk.word_tokenize(text) if not t.isalnum()]


class PosDistribution:
    """Obtains the pos tag distribution for inputted text"""

    def __init__(self):
        self.keys = {}
        self._keys = list(nltk.load('help/tagsets/upenn_tagset.pickle').keys())

        for key in self._keys:
            self.keys[key] = 0

    def __call__(self, text):
        tokens = nltk.word_tokenize(text)
        counts = Counter(tag for _, tag in nltk.pos_tag(tokens))
        count_dict = self.keys.copy()
        for tag, count in counts.items():
            count_dict[tag] = count

        return count_dict
