import csv
import glob
import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Union

import nltk
from pybrat.parser import BratParser
from sklearn.model_selection import train_test_split

import canary.utils
from canary import logger
from canary.corpora.araucaria import Nodeset, Edge, Locution, Node
from canary.corpora.essay_corpus import find_paragraph_features, find_cover_sentence_features, find_cover_sentence, \
    tokenize_essay_sentences, find_component_position
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
                logging.error("There was an error fetching the corpus")

        elif corpora_already_downloaded:
            logger.info(f"Corpus already present at {storage_location}")
            return {
                "corpus": corpora,
                "location": storage_location
            }


def load_essay_corpus(purpose=None, merge_premises=False, version=2, train_split_size=0.5, **kwargs):
    """
    Loads essay corpus version

    :param train_split_size:
    :param purpose:
    :param merge_premises: whether or not to combine claims and major claims
    :param version: d
    :return:
    """

    _allowed_purpose_values = [
        None,
        'component_prediction',
        'relation_prediction',
        'sequence_labelling'
    ]

    canary.utils.nltk_download(['punkt'])
    _allowed_version_values = [1, 2, "both"]

    if train_split_size is not None:
        if not (0 < train_split_size < 1):
            raise ValueError("Split value should be between 0 and 1.")

    if version not in _allowed_version_values:
        raise ValueError(f"{version} is not a valid value. Valid values are {_allowed_version_values}")

    if purpose not in _allowed_purpose_values:
        raise ValueError(f"{purpose} is not a valid value. Valid values are {_allowed_purpose_values}")

    def get_corpus(v):
        essay_corpus_location = Path(CANARY_CORPORA_LOCATION) / "brat-project-final" if v == 2 else Path(
            CANARY_CORPORA_LOCATION) / "brat-project"
        if os.path.exists(essay_corpus_location) is False:
            corpus = download_corpus(f"argument_annotated_essays_{v}")
            zip_name = "brat-project-final" if v == 2 else "brat-project"
            corpus_zip = corpus['location'] / f"ArgumentAnnotatedEssays-{v}.0/{zip_name}.zip"
            with zipfile.ZipFile(corpus_zip) as z:
                z.extractall(CANARY_CORPORA_LOCATION)

        brat_parser = BratParser(error="ignore")
        e = brat_parser.parse(essay_corpus_location)
        return e

    if version in [1, 2]:
        essays = get_corpus(version)
    else:
        essay_corpus_1 = get_corpus(1)
        essay_corpus_2 = get_corpus(2)
        essays = essay_corpus_1 + essay_corpus_2

    if purpose is None:
        return essays

    elif purpose == "component_prediction":
        X, Y = [], []
        for essay in essays:
            essay.sentences = tokenize_essay_sentences(essay)
            for entity in essay.entities:
                component_feats = {
                    "component": entity.mention,
                    "cover_sentence": find_cover_sentence(essay, entity)
                }
                component_feats.update(find_component_position(essay, entity))
                X.append(component_feats)
                if merge_premises is False:
                    Y.append(entity.type)
                else:
                    if entity.type == "MajorClaim":
                        Y.append("Claim")
                    else:
                        Y.append(entity.type)

        train_data, test_data, train_targets, test_targets = \
            train_test_split(X, Y,
                             train_size=train_split_size,
                             shuffle=True,
                             random_state=0
                             )

        return train_data, test_data, train_targets, test_targets

    elif purpose == "relation_prediction":
        X = []
        Y = []

        for essay in essays:

            for index, relation in enumerate(essay.relations):
                features = {
                    "essay_id": essay.id,
                    "arg1_text": relation.arg1.mention,
                    "arg1_type": relation.arg1.type,
                    "arg1_start": relation.arg1.start,
                    "arg1_end": relation.arg1.end,
                    "arg2_text": relation.arg2.mention,
                    "arg2_type": relation.arg2.type,
                    "arg2_start": relation.arg2.start,
                    "arg2_end": relation.arg2.end,
                    "n_components_in_essay": len(essay.relations),
                }

                find_cover_sentence_features(features, essay, relation)

                find_paragraph_features(features, relation, essay)

                X.append(features)
                Y.append(relation.type)

        train_data, test_data, train_targets, test_targets = \
            train_test_split(X, Y,
                             train_size=train_split_size,
                             shuffle=True,
                             random_state=0,
                             )
        return train_data, test_data, train_targets, test_targets

    elif purpose == "sequence_labelling":
        X = []
        Y = []

        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentence_tokenizer._params.abbrev_types.update(['i.e', "e.g", "etc"])

        for _essay in essays:
            logger.debug(f"Reading {_essay.id}")
            # keep track of number of entities found
            num_f = 0
            args = []

            # tokenise essay
            _essay.sentences = tokenize_essay_sentences(_essay)
            essay_tokens = _essay.sentences

            #  sort entities by starting position in the text
            entities = sorted(_essay.entities, key=lambda x: x.start)

            # initialise
            x1 = [nltk.word_tokenize(tokens) for tokens in essay_tokens]
            y1 = [["O" for _ in tokens] for tokens in x1]

            # Deal with a few formatting / misc errors to get data into right shape

            if _essay.id == 'essay098':
                entities[4].mention = 'when children take jobs, they tend to be more responsible'

            if _essay.id == "essay114":
                entities[8].mention += "n"

            if _essay.id == "essay182":
                entities[19].mention += "t"

            if _essay.id == "essay248":
                entities[1].mention += "t"

            if _essay.id == "essay330":
                x1[5][15] = "doing"
                x1[5].insert(16, ".")
                x1[5].insert(17, "In")
                y1[5].append("O")
                y1[5].append("O")

            if _essay.id == "essay337":
                entities[11].mention += "n"

            # get first entity to look for
            current_ent = entities.pop(0)

            # look through each sentence
            while 0 <= num_f < len(_essay.entities):
                for i in range(len(essay_tokens)):
                    ent_tokens = nltk.word_tokenize(current_ent.mention)

                    try:
                        # check if we have all the elements we need
                        if all(e in x1[i] for e in ent_tokens) is True:

                            # navigate through sentence looking for arg component span
                            for j in range(len(x1[i])):
                                matches = []

                                # look through sentence from this position and see if it matches the tokens in ent
                                # start: j
                                # end: ent_token end...

                                l: int = 0
                                for k in range(j, len(x1[i])):
                                    try:
                                        if 0 <= k < len(x1[i]) and 0 <= l < len(ent_tokens):
                                            if x1[i][k] == ent_tokens[l]:
                                                matches.append((x1[i][k], True, k))
                                                l = l + 1
                                            else:
                                                matches.append((x1[i][k], False, k))
                                    except IndexError as e:
                                        logger.error(e)

                                # we have an argumentative match
                                if all(x[1] is True for x in matches) and matches != []:
                                    num_f += 1
                                    args.append([m[0] for m in matches])
                                    # get next entity to search for
                                    if len(entities) > 0:
                                        current_ent = entities.pop(0)

                                    for index, m in enumerate(matches):
                                        if index == 0:
                                            y1[i][m[2]] = "Arg-B"
                                        else:
                                            y1[i][m[2]] = "Arg-I"
                    except IndexError as e:
                        logger.error(e)

            if [nltk.word_tokenize(m.mention) for m in sorted(_essay.entities, key=lambda x: x.start)] == args is False:
                raise ValueError("Something went wrong when getting corpora")

            if num_f != len(_essay.entities):
                logger.warn(ValueError(
                    f"Did not find all the argument components on {_essay.id}. {num_f} / {len(_essay.entities)}"))

            if num_f > len(_essay.entities):
                # essay186
                logger.warn("...")

            else:
                X = x1 + X
                Y = y1 + Y

        # If data and label shapes are not the same, the algorithm will not work.
        # Check this ahead of time
        errors = 0
        for j in zip(X, Y):
            if len(j[0]) != len(j[1]):
                errors += 1

        if errors > 0:
            raise ValueError(f'Data is incorrect shape. Number of errors {errors}')

        train_data, test_data, train_targets, test_targets = \
            train_test_split(X, Y,
                             train_size=train_split_size,
                             shuffle=True,
                             random_state=0,
                             )

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

    except FileNotFoundError:
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
    corpus = {}
    if "location" in corpus_download:

        corpus_location = corpus_download["location"]
        files = glob.glob(str(Path(corpus_location) / "nodeset*.json"))

        for file in files:
            file = Path(file)
            node = Nodeset()
            with open(file, "r", encoding="utf8") as json_file:
                node.id = file.stem
                j = json.load(json_file)
                node.load(j)
            json_file.close()

            with open(str(Path(corpus_location / f"{file.stem}.txt")), "r", encoding="utf8") as text_file:
                node.text = text_file.read()
            text_file.close()
            corpus[node.id] = node

        if purpose is None:
            return corpus

    return corpus
