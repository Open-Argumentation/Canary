import csv
import glob
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Union

from pybrat.parser import BratParser
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

        corpora_already_downloaded = os.path.isdir(file)

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
            logger.info(f"Corpus already present at {storage_location}")
            return {
                "corpus": corpora,
                "location": storage_location
            }


def load_essay_corpus(purpose=None, merge_premises=False):
    """
    Loads essay corpus version 2

    :param purpose:
    :param merge_premises: whether or not to combine claims and major claims
    :return:
    """

    _allowed_purpose_values = [
        None,
        'component_prediction',
        'relation_prediction'
    ]

    if purpose not in _allowed_purpose_values:
        raise ValueError(f"{purpose} is not a valid value. Valid values are {_allowed_purpose_values}")

    essay_corpus_location = Path(CANARY_CORPORA_LOCATION) / "brat-project-final"

    if os.path.exists(essay_corpus_location) is False:
        corpus = download_corpus("argument_annotated_essays_2")
        corpus_zip = corpus['location'] / "ArgumentAnnotatedEssays-2.0/brat-project-final.zip"
        with zipfile.ZipFile(corpus_zip) as z:
            z.extractall(CANARY_CORPORA_LOCATION)

    brat_parser = BratParser(error="ignore")
    essays = brat_parser.parse(essay_corpus_location)

    if purpose is None:
        return essays

    elif purpose == "component_prediction":
        X, Y = [], []
        for essay in essays:
            for entity in essay.entities:
                X.append(entity.mention)
                if merge_premises is False:
                    Y.append(entity.type)
                else:
                    if entity.type == "MajorClaim":
                        Y.append("Claim")
                    else:
                        Y.append(entity.type)

        train_data, test_data, train_targets, test_targets = \
            train_test_split(X, Y,
                             train_size=0.9,
                             shuffle=True,
                             random_state=0
                             )

        return train_data, test_data, train_targets, test_targets

    elif purpose == "relation_prediction":

        X = []
        Y = []

        for essay in essays:
            for relation in essay.relations:
                X.append({
                    "arg1_text": relation.arg1.mention,
                    "arg1_type": relation.arg1.type,
                    "arg1_start": relation.arg1.start,
                    "arg1_end": relation.arg1.end,
                    "arg2_text": relation.arg2.mention,
                    "arg2_type": relation.arg2.type,
                    "arg2_start": relation.arg2.start,
                    "arg2_end": relation.arg2.end,
                })
                Y.append(relation.type)

        train_data, test_data, train_targets, test_targets = \
            train_test_split(X, Y,
                             train_size=0.9,
                             shuffle=True,
                             random_state=0,
                             # stratify=Y
                             )
        return train_data, test_data, train_targets, test_targets
        # train_features, test_features, test_labels, train_labels = [], [], [], []
        # ss = StratifiedKFold(n_splits=10, shuffle=False,)
        #
        # X = numpy.array(X)
        # Y = numpy.array(Y)
        # for train_index, test_index in ss.split(X, Y):
        #     train_features, test_features = X[train_index], X[test_index]
        #     train_labels, test_labels = Y[train_index], Y[test_index]
        # return train_features, test_features, train_labels, test_labels


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
            train_data.append({"text": row[2], "topic": row[1]})
            train_targets.append(int(row[4]))

    with open(Path(
            f'{ROOT_DIR}/data/datasets/ibm/IBM_debater_evidence_sentences/test.csv'), encoding="utf8") as data:
        csv_reader = csv.reader(data)
        next(csv_reader)
        for row in csv_reader:
            test_data.append({"text": row[2], "topic": row[1]})
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


def load_araucaria_corpus(purpose: str = None):
    """
    Loads the araucaria corpus

    :return: the corpus
    """

    corpus_download = download_corpus("araucaria")
    corpus = None
    corpus_entries = []
    if "location" in corpus_download:

        corpus_location = corpus_download["location"]
        files = glob.glob(str(Path(corpus_location) / "nodeset*.json"))

        for file in files:
            file = Path(file)
            entry = {}
            with open(file, "r", encoding="utf8") as json_file:
                entry["nodeset"] = file.stem
                entry["json"] = json.load(json_file)
            json_file.close()
            with open(str(Path(corpus_location / f"{file.stem}.txt")), "r", encoding="utf8") as text_file:
                entry["text"] = text_file.read()
            text_file.close()

            corpus_entries.append(entry)

        if purpose is None:
            corpus = corpus_entries
            return corpus
    elif purpose == "scheme_prediction":
        train_data = []
        test_data = []
        train_target = []
        test_target = []

        for entry in corpus_entries:
            pass

    return corpus
