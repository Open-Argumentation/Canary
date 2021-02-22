import os
import nltk
from configparser import ConfigParser
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer


class Vectorizer:

    def vectoriser(self, ngram_range=(2, 3), apply_lemmatization=False):
        return CountVectorizer(ngram_range=ngram_range, stop_words=Preprocessor().stopwords,
                               )


class Lemmatizer:

    def __init__(self):
        self.word_net = nltk.WordNetLemmatizer()

    def __call__(self, text):
        return [self.word_net.lemmatize(t) for t in nltk.word_tokenize(text)]


class Preprocessor:

    def __init__(self):
        __config = ConfigParser()
        __config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'etc/canary.cfg'))
        nltk_data_directory = os.path.join(Path.home(), __config.get('nltk', 'storage_directory'))
        nltk.data.path.append(nltk_data_directory)
        nltk.download(['stopwords', 'punkt', 'wordnet'], download_dir=nltk_data_directory, quiet=True)

    __stopwords = [
        ",",
        "br",
        "also",
        "'d",
        "'ll",
        "'re",
        "'s",
        "'ve",
        'could',
        'doe',
        'ha',
        'might',
        'must',
        "n't",
        'need',
        'sha',
        'wa',
        'wo',
        'would',
    ]

    @property
    def stopwords(self) -> list:
        sw = nltk.corpus.stopwords.words('english') + self.__stopwords
        return sw

    def extract_bigrams(self, sentences) -> list:
        """
        A function to extract bigrams from a corpus

        :param sentences: a list of reviews
        :returns list: a list of bigrams
        """

        all_bigrams = []
        for sentence in sentences:
            token = nltk.word_tokenize(sentence)
            bigrams = nltk.bigrams(token)
            for bigram in bigrams:
                w1, w2 = bigram
                if w1 not in self.stopwords and w2 not in self.stopwords:
                    if w1.isalpha() and w2.isalpha():
                        all_bigrams.append(bigram)
        return all_bigrams

    def extract_unigram(self, sentences) -> list:
        """
        A function to extract unigrams from a corpus

        :param sentences: a list of reviews
        :returns list: a list of unigrams
        """

        tokens = []
        for sentence in sentences:
            token = nltk.word_tokenize(sentence)
            for t in token:
                if t not in self.stopwords and t.isalpha():
                    tokens.append(t)
        return tokens

    def filter_stopwords(self, sentences: list, stopwords: list) -> list:
        pass

    def extract_ngrams(self) -> list:
        pass
