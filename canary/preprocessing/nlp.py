from pathlib import Path as _Path


def nltk_download(packages):
    """Wrapper around nltk.download """
    import nltk
    import os
    from ..utils import config as _config
    from ..utils import CANARY_LOCAL_STORAGE
    from ..utils import logger as _logger

    nltk_data_dir = _Path(f"{_Path.home()}") / _config.get("nltk", "storage_directory")
    os.makedirs(CANARY_LOCAL_STORAGE, exist_ok=True)
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
                # elif package == "tagsets":
                #     nltk.data.find("tagsets/tagsets")
            except LookupError:
                _logger.debug(f"Didn't find {package}. Attempting download.")
                nltk.download(package, quiet=True, download_dir=nltk_data_dir)


def spacy_download(package: str = None, disable=None):
    """
    Wrapper around spacy.load which will download the necessary package if it is not present
    :return:spacy model
    """

    from spacy import load
    from spacy.cli import download
    from ..utils import logger as _logger
    from ..utils import config as _config

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
        if disable is None:
            nlp = load(package)
        else:
            nlp = load(package, disable=disable)

        if disable is not None:
            if 'benepar' not in disable:
                nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        else:
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})

        return nlp
    except OSError:
        try:
            _logger.debug(f"Did not find spacy model {package}. Downloading now.")
            download(package)
            return load(package)
        except OSError:
            _logger.error(f"could not download {package}")
