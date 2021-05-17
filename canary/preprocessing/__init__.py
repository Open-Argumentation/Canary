from canary import config
import os
import nltk
from pathlib import Path


class Preprocessor:

    def __init__(self):
        nltk_data_directory = os.path.join(Path.home(), config.get('nltk', 'storage_directory'))
        nltk.data.path.append(nltk_data_directory)
        nltk.download(['stopwords', 'punkt'],
                      download_dir=nltk_data_directory, quiet=True)

    __stopwords = [
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
    ]

    @property
    def stopwords(self) -> list:
        """

        :return: a list of stopwords
        """
        sw = nltk.corpus.stopwords.words('english') + self.__stopwords
        return sw


class Lemmatizer:

    def __init__(self):
        self.word_net = nltk.WordNetLemmatizer()

    def __call__(self, text):
        return [self.word_net.lemmatize(t) for t in nltk.word_tokenize(text)]