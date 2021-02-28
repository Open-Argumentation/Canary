import os
import nltk
from configparser import ConfigParser
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessor:

    def __init__(self):
        __config = ConfigParser()
        __config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'etc/canary.cfg'))
        nltk_data_directory = os.path.join(Path.home(), __config.get('nltk', 'storage_directory'))
        nltk.data.path.append(nltk_data_directory)
        nltk.download(['stopwords', 'punkt'],
                      download_dir=nltk_data_directory, quiet=True)

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
