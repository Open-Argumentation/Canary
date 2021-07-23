import nltk
import spacy

nlp = spacy.load("en_core_web_lg")

class Lemmatizer:
    """
    Transforms text into its lemma form

    e.g.
    - cats -> cat
    - corpora -> corpus
    """

    def __init__(self):
        self.word_net = nltk.WordNetLemmatizer()

    def __call__(self, text):
        return [self.word_net.lemmatize(t) for t in nltk.word_tokenize(text)]


class PosLemmatizer:

    def t(self, x):
        return f"{x.lemma_}/{x.tag_}"

    def __call__(self, text):
        text = nlp(text)
        return [self.t(d) for d in text]


class Stemmer:
    """
    Transforms text into its stemmed form
    """

    def __init__(self):
        self.stemmer = nltk.PorterStemmer()

    def __call__(self, text):
        return [self.stemmer.stem(token) for token in nltk.word_tokenize(text)]


class PunctuationTokenizer:
    """
    Extracts only punctuation from a piece of text

    e.g.
    - Hi, what's up? Did you like the movie last night?
    -> [[','], ["'", 's'], ['?'], ['?']]
    """

    def __init__(self):
        self.tokenizer = nltk.WordPunctTokenizer()

    def __call__(self, text):
        return [self.tokenizer.tokenize(t) for t in nltk.word_tokenize(text) if not t.isalnum()]
