from pathlib import Path as _Path

import canary as _canary
from canary.utils import config as _config

_spacy = []
_nltk = []


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
            if package in _nltk:
                _canary.utils.logger.debug("nltk package loaded previously.")
                return
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
                _canary.utils.logger.debug(f"Didn't find {package}. Attempting download.")
                nltk.download(package, quiet=True, download_dir=nltk_data_dir)
                _nltk.append(package)


def spacy_download(package: str = None):
    """
    Wrapper around spacy.load which will download the necessary package if it is not present
    :return:spacy model
    """
    for spacy_model in _spacy:
        if spacy_model[0] == package:
            _canary.utils.logger.debug("Spacy model already loaded.")
            return spacy_model[1]

    _canary.utils.logger.debug("spacy loading")
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

        _spacy.append((package, nlp))
        return nlp
    except OSError:
        try:
            _canary.utils.logger.debug(f"Did not find spacy model {package}. Downloading now.")
            download(package)
            return load(package)
        except OSError:
            _canary.utils.logger.error(f"could not download {package}")
