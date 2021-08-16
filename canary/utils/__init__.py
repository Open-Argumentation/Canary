import os as _os
from pathlib import Path as _Path

import canary
from canary import config as _config

# Various utility variables
CANARY_ROOT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(canary.__file__)))
CONFIG_LOCATION = _Path(f"{CANARY_ROOT_DIR}/etc") / "canary.cfg"
CANARY_LOCAL_STORAGE = _Path(f"{_Path.home()}/{_config.get('canary', 'storage_location')}")
CANARY_MODEL_STORAGE_LOCATION = _Path(f"{CANARY_LOCAL_STORAGE}/models")
CANARY_MODEL_DOWNLOAD_LOCATION = _config.get('canary', 'model_download_location')
CANARY_CORPORA_LOCATION = _os.path.join(_Path.home(), _config.get('canary',
                                                                  'corpora_home_storage_directory'))


# Utility functions

def nltk_download(packages):
    """Wrapper around nltk.download """
    import nltk
    nltk_data_dir = _Path(f"{_Path.home()}") / _config.get("nltk", "storage_directory")

    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    if type(packages) is str:
        packages = [packages]

    if type(packages) is list:
        for package in packages:
            try:
                if package == "punkt":
                    nltk.data.find("tokenizers/punkt")
                elif package == "averaged_perceptron_tagger":
                    nltk.data.find("taggers/averaged_perceptron_tagger")
                elif package == "wordnet":
                    nltk.data.find("corpora/wordnet")
                elif package == "maxent_ne_chunker":
                    nltk.data.find("chunkers/maxent_ne_chunker")
                elif package == "words":
                    nltk.data.find("corpora/words")
            except LookupError:
                canary.logger.debug(f"Didn't find {package}. Attempting download.")
                nltk.download(package, quiet=True, download_dir=nltk_data_dir)


def spacy_download(package: str = None):
    """
    Wrapper around spacy.load which will download the necessary package if it is not present
    :return:spacy model
    """

    from spacy import load
    from spacy.cli import download
    import benepar
    import nltk

    nltk_data_dir = _Path(f"{_Path.home()}") / _config.get("nltk", "storage_directory")

    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    if package is None:
        package = "en_core_web_lg"
    try:
        nltk_data_dir = _Path(f"{_Path.home()}") / _config.get("nltk", "storage_directory")
        benepar.download('benepar_en3', download_dir=nltk_data_dir, quiet=True)
        nlp = load(package)
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

        return nlp
    except OSError:
        try:
            canary.logger.debug(f"Did not find spacy model {package}. Downloading now.")
            download(package)
            return load(package)
        except OSError:
            canary.logger.error(f"could not download {package}")
