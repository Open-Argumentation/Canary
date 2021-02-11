import os
from configparser import ConfigParser
from pathlib import Path
from urllib.error import HTTPError


def download_corpus(corpus_id: str, overwrite_existing: bool = False, save_location: str = None):
    """
    Downloads a corpus to be used for argumentation mining.

    :param str save_location: the absolute path to the directory where the corpus should be saved
    :param str corpus_id: the idea of the corpus which corresponds to the id in data/corpus.json
    :param bool overwrite_existing: should Canary overwrite an existing corpus if it has already been downloaded?
    """

    config = ConfigParser()
    config.read('../etc/canary.cfg')
    storage_location = \
        os.path.join(Path.home(), config.get('canary',
                                             'corpora_home_storage_directory')) if save_location is None else save_location
    storage_location_tarfile = f'{storage_location}/{corpus_id}.tar.gz'
    storage_location = os.path.join(storage_location, corpus_id)

    with open("../data/corpora.json") as corpora:
        import json
        import tarfile

        corpora = json.load(corpora)
        corpora = [corpus for corpus in corpora if corpus_id == corpus['id']]
        if len(corpora) == 1:
            corpora = corpora[0]
        else:
            raise ValueError('Invalid corpus id.')

        corpora_already_downloaded = os.path.isfile(storage_location_tarfile)

        if corpora and corpora_already_downloaded is False or corpora and overwrite_existing is True:
            import requests
            response = requests.get(corpora["download_url"], stream=True)
            if response.status_code == 200:
                os.makedirs(storage_location, exist_ok=True)
                with open(storage_location_tarfile, "wb") as f:
                    f.write(response.raw.read())
                    f.close()
                    tf = tarfile.open(storage_location_tarfile)
                    tf.extractall(path=f"{os.path.join(storage_location, corpus_id)}")
                    tf.close()
                    print(f"Corpus downloaded to {storage_location}")
            else:
                raise HTTPError(
                    f"There was an error fetching the requested resource. HTTP status code {response.status_code}")
        elif corpora_already_downloaded:
            print(f"Corpus already present at {storage_location}")
