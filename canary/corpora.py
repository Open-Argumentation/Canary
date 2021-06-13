import csv
import glob
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Union

from sklearn.model_selection import train_test_split

from canary import logger
from canary.utils import ROOT_DIR, CANARY_CORPORA_LOCATION


def download_corpus(corpus_id: str, overwrite_existing: bool = False, save_location: str = None) -> dict:
    """
    Downloads a corpus to be used for argumentation mining.

    :param str save_location: the absolute path to the directory where the corpus should be saved
    :param str corpus_id: the idea of the corpus which corresponds to the id in data/corpus.json
    :param bool overwrite_existing: should Canary overwrite an existing corpus if it has already been downloaded?
    """

    os.makedirs(CANARY_CORPORA_LOCATION, exist_ok=True)
    storage_location = CANARY_CORPORA_LOCATION if save_location is None else save_location
    file = f'{storage_location}/{corpus_id}'
    storage_location = Path(f"{storage_location}/{corpus_id}")

    with open(f"{ROOT_DIR}/data/corpora.json") as corpora:
        corpora = json.load(corpora)
        corpora = [corpus for corpus in corpora if corpus_id == corpus['id']]
        if len(corpora) == 1:
            corpora = corpora[0]
        else:
            raise ValueError('Invalid corpus id.')

        corpora_already_downloaded = os.path.isfile(file)

        if corpora and corpora_already_downloaded is False or corpora and overwrite_existing is True:
            import requests
            try:
                response = requests.get(corpora["download_url"], stream=True)
                if response.status_code == 200:
                    type = response.headers.get('Content-Type')
                    if type == 'application/zip':
                        file = file + ".zip"
                    elif type == "application/tar.gz":
                        file = file + ".tar.gz"

                    with open(file, "wb") as f:
                        f.write(response.raw.read())
                        f.close()
                    if type == "application/tar.gz":
                        tf = tarfile.open(f"{storage_location}.tar.gz", "r")
                        tf.extractall(f"{CANARY_CORPORA_LOCATION}/{corpus_id}")
                        tf.close()
                    elif type == "application/zip":
                        with zipfile.ZipFile(f"{storage_location}.zip", "r") as zf:
                            zf.extractall(f"{CANARY_CORPORA_LOCATION}/{corpus_id}")

                    logger.info(f"Corpus downloaded to {storage_location}")
                    return {
                        "corpus": corpora,
                        "location": storage_location
                    }
            except:
                print("There was an error fetching the corpus")

        elif corpora_already_downloaded:
            print(f"Corpus already present at {storage_location}")


def load_essay_corpus(merge_premises=False) -> tuple:
    """
    Loads essay corpus version 2

    :param merge_premises: whether or not to combine claims and major claims
    :return: the corpus as a tuple
    """

    essay_corpus_location = Path(CANARY_CORPORA_LOCATION) / "brat-project-final"
    if os.path.exists(essay_corpus_location) is False:
        corpus = download_corpus("argument_annotated_essays_2")
        corpus_zip = corpus['location'] / "ArgumentAnnotatedEssays-2.0/brat-project-final.zip"
        with zipfile.ZipFile(corpus_zip) as z:
            z.extractall(CANARY_CORPORA_LOCATION)

    documents = []
    os.chdir(essay_corpus_location)
    for file in glob.glob("essay*.ann"):
        file_data = []
        with open(file, encoding="utf8") as ann:
            csv_reader = csv.reader(ann, delimiter="\t")
            for row in csv_reader:
                file_data.append(row)
        documents.append(file_data)

    X, Y = [], []
    for doc in documents:
        for line in doc:
            if 2 < len(line):
                component = str.split(line[1])[0]
                if component != 'supports' and component != 'Stance' and component != 'attacks':
                    X.append(line[2])
                    if merge_premises is True and component == 'MajorClaim':
                        Y.append("Claim")
                    else:
                        Y.append(component)

    train_data, test_data, train_targets, test_targets = train_test_split(X, Y, train_size=0.9, shuffle=True,
                                                                          random_state=0)

    return train_data, test_data, train_targets, test_targets


def load_imdb_debater_evidence_sentences() -> tuple:
    """
    Load the imdb debater corpus

    :return: the corpus as a tuple
    """

    train_data, test_data, train_targets, test_targets = [], [], [], []

    with open(Path(
            f'{ROOT_DIR}/data/datasets/ibm/IBM_debater_evidence_sentences/train.csv'), encoding="utf8") as data:
        csv_reader = csv.reader(data)
        next(csv_reader)
        for row in csv_reader:
            train_data.append(row[2])
            train_targets.append(int(row[4]))

    with open(Path(
            f'{ROOT_DIR}/data/datasets/ibm/IBM_debater_evidence_sentences/test.csv'), encoding="utf8") as data:
        csv_reader = csv.reader(data)
        next(csv_reader)
        for row in csv_reader:
            test_data.append(row[2])
            test_targets.append(int(row[4]))

    return train_data, train_targets, test_data, test_targets


def load_ukp_sentential_argument_detection_corpus(multiclass=True) -> Union[list, dict]:
    """
    Load the ukp sentential argument corpus

    :param multiclass: whether to return a multiclass problem
    :return: the dataset
    """

    dataset_format = {'train': [], "test": [], "val": []}
    datasets = {
        "nuclear_energy": dataset_format,
        "death_penalty": dataset_format,
        "minimum_wage": dataset_format,
        "marijuana_legalization": dataset_format,
        "school_uniforms": dataset_format,
        "gun_control": dataset_format,
        "abortion": dataset_format,
        "cloning": dataset_format,
    }

    def multiclass_transformer(val):
        if multiclass is False:
            if val == 'NoArgument':
                return False
            else:
                return True
        else:
            return val

    try:
        for d in datasets:
            file = Path(f"{ROOT_DIR}/data/datasets/ukp/ukp_sentential_argument_mining_corpus/data/complete/{d}.tsv",
                        encoding="utf8")

            if os.path.isfile(file) is False:
                raise FileNotFoundError(f"{d}.tsv from the UKP dataset does not exist. Has it been downloaded?")

            with open(file, encoding="utf8") as data:
                csv_reader = csv.reader(data, delimiter="\t", quotechar='"')
                next(csv_reader)  # skip heading

                for row in csv_reader:
                    if 0 <= 6 < len(row):
                        if row[6] == 'train':
                            datasets[d]['train'].append([row[4], multiclass_transformer(row[5])])
                        elif row[6] == 'test':
                            datasets[d]['test'].append([row[4], multiclass_transformer(row[5])])
                        else:
                            datasets[d]['val'].append([row[4], multiclass_transformer(row[5])])

    except FileNotFoundError as e:
        logger.error("Could not find a file from the UKP sentential dataset. Please ensure it has been downloaded.")
    except Exception as e:
        logger.error(e)
    finally:
        return datasets
