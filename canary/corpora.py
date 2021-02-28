import csv
import os
from configparser import ConfigParser
from pathlib import Path
from urllib.error import HTTPError
from canary.utils import ROOT_DIR
from canary import logger


def download_corpus(corpus_id: str, overwrite_existing: bool = False, save_location: str = None):
    """
    Downloads a corpus to be used for argumentation mining.

    :param str save_location: the absolute path to the directory where the corpus should be saved
    :param str corpus_id: the idea of the corpus which corresponds to the id in data/corpus.json
    :param bool overwrite_existing: should Canary overwrite an existing corpus if it has already been downloaded?
    """

    config = ConfigParser()
    config.read(f"{ROOT_DIR}/etc/canary.cfg")
    storage_location = \
        os.path.join(Path.home(), config.get('canary',
                                             'corpora_home_storage_directory')) if save_location is None else save_location
    storage_location_tarfile = f'{storage_location}/{corpus_id}.tar.gz'
    storage_location = os.path.join(storage_location, corpus_id)

    with open(f"{ROOT_DIR}/data/corpora.json") as corpora:
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


def load_imdb_debater_evidence_sentences() -> tuple:
    train_data, test_data, train_targets, test_targets = [], [], [], []

    with open(
            f'{ROOT_DIR}/data/datasets/ibm/IBM_debater_evidence_sentences/train.csv') as data:
        csv_reader = csv.reader(data)
        next(csv_reader)
        for row in csv_reader:
            train_data.append(row[2])
            train_targets.append(int(row[4]))

    with open(
            f'{ROOT_DIR}/data/datasets/ibm/IBM_debater_evidence_sentences/test.csv') as data:
        csv_reader = csv.reader(data)
        next(csv_reader)
        for row in csv_reader:
            test_data.append(row[2])
            test_targets.append(int(row[4]))

    return train_data, train_targets, test_data, test_targets


def load_ukp_sentential_argument_detection_corpus() -> dict:
    datasets = {
        "nuclear_energy": {'train': [], "test": [], "val": []},
        "death_penalty": {'train': [], "test": [], "val": []},
        "minimum_wage": {'train': [], "test": [], "val": []},
        "marijuana_legalization": {'train': [], "test": [], "val": []},
        "school_uniforms": {'train': [], "test": [], "val": []},
        "gun_control": {'train': [], "test": [], "val": []},
        "abortion": {'train': [], "test": [], "val": []},
        "cloning": {'train': [], "test": [], "val": []},
    }
    try:
        for d in datasets:
            file = os.path.join(f"{ROOT_DIR}/data/datasets/ukp/ukp_sentential_argument_mining_corpus/data/complete/",
                                f"{d}.tsv")

            if os.path.isfile(file) is False:
                raise FileNotFoundError(f"{d}.tsv from the UKP dataset does not exist. Has it been downloaded?")

            with open(file) as data:
                csv_reader = csv.reader(data, delimiter="\t", quotechar='"')
                next(csv_reader)  # skip heading

                for row in csv_reader:
                    if 0 <= 6 < len(row):
                        if row[6] == 'train':
                            datasets[d]['train'].append([row[4], row[5]])
                        elif row[6] == 'test':
                            datasets[d]['test'].append([row[4], row[5]])
                        else:
                            datasets[d]['val'].append([row[4], row[5]])
    except FileNotFoundError as e:
        logger.error("Could not find a file from the UKP sentential dataset. Please ensure it has been downloaded.")
    except Exception as e:
        logger.error(e)
    finally:
        return datasets
