"""Corpora Package"""
import glob
import itertools
import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Union

import nltk
from pybrat.parser import BratParser

from ..nlp._utils import nltk_download
from ..utils import CANARY_ROOT_DIR, CANARY_CORPORA_LOCATION, logger

__all__ = [
    "download_corpus",
    "load_corpus",
    "load_essay_corpus",
    "load_araucaria_corpus"
]


def load_corpus(corpus_id: str, download_if_missing=False) -> Optional[list]:
    """Loads a corpus that has previously been downloaded

    Parameters
    ----------
    corpus_id: str
        The id of the corpus to load.
    download_if_missing: bool, False
        If the corpus is not present on disk, should Canary attempt to download it?

    Returns
    -------
    Union[list, None]
        if a corpus can be loaded, a list of relevant dataset files will be returned. Otherwise nothing will be
        returned.

    Raises
    -------
    UserWarning
        A warning is raised if the requested corpus cannot be found.
    """
    allowed_values = [x.stem for x in Path(CANARY_CORPORA_LOCATION).iterdir() if x.is_dir()]

    with open(f"{CANARY_ROOT_DIR}/_data/corpora.json") as corpora:
        corpora = json.load(corpora)
        corpora_ids = [corpus['id'] for corpus in corpora]
        allowed_values += corpora_ids

    if corpus_id not in allowed_values:
        raise ValueError(f"Incorrect corpus id supplied. Allowed values are: {allowed_values}")

    # Corpus id should now be valid and will be in corpora root if downloaded
    corpus_location = Path(CANARY_CORPORA_LOCATION) / corpus_id
    if os.path.isdir(corpus_location) is False and download_if_missing is True:
        download_corpus(corpus_id)
        return load_corpus(corpus_id)
    if os.path.isdir(corpus_location) is False and download_if_missing is False:
        raise UserWarning("It appears the requested corpus has not been downloaded and is not present on disk. "
                          "Have you downloaded it? You can set download_if_missing to True and the "
                          "corpus will be downloaded. Alternatively, use the function download_corpus.")
    import glob
    return glob.glob(f"{corpus_location}/*")


def download_corpus(corpus_id: str, overwrite_existing: bool = False, save_location: str = None,
                    aifdb_corpus=False, request_timeout=60) -> dict:
    """Downloads a corpus to be used for argumentation mining.

    Parameters
    ----------
    corpus_id: str
        the absolute path to the directory where the corpus should be saved
    overwrite_existing: bool, default=False
        Should the corpus be overwritten if already present?
    save_location: str, optional
        Where the corpus should be downloaded to. Defaults to the canary corpora directory.
    aifdb_corpus: bool, False
        A boolean value indicating if the corpus should be fetched from aifdb.
    request_timeout: int, 60
        How long Canary wait when trying to download a corpus from a remote resource such as aifdb.

    Notes
    ------
    If aifdb_corpus is set to true, the corpora will be downloaded directly from aifdb.org.
    These corpora are provided at the discretion of the site owners and can disappear / be altered at anytime.

    Returns
    -------
    dict
        The details of the corpus provided as a dictionary
    """

    os.makedirs(CANARY_CORPORA_LOCATION, exist_ok=True)
    storage_location = CANARY_CORPORA_LOCATION if save_location is None else save_location
    file = f'{storage_location}/{corpus_id}'
    storage_location = Path(f"{storage_location}/{corpus_id}")

    corpora = {}
    if aifdb_corpus is False:
        with open(f"{CANARY_ROOT_DIR}/_data/corpora.json") as corpora:
            corpora = json.load(corpora)
            corpora_ids = [corpus['id'] for corpus in corpora]
            corpora = [corpus for corpus in corpora if corpus_id == corpus['id']]
            if len(corpora) == 1:
                corpora = corpora[0]
            else:
                raise ValueError(
                    f'Invalid corpus id. Allowed values are: {corpora_ids}. '
                    'If the corpus is an aifdb corpus, set aif_corpus to True.')
    else:
        corpora['download_url'] = f"http://corpora.aifdb.org/zip/{corpus_id}/download"
        corpora['id'] = corpus_id

    corpora_already_downloaded = os.path.isdir(file)

    if corpora and corpora_already_downloaded is False or corpora and overwrite_existing is True:
        import requests
        try:
            if aifdb_corpus is True and request_timeout is None:
                logger.warn(
                    "Setting request_timeout to None is not recommended. "
                    "The request may never finish if the server does not respond.")
            response = requests.get(corpora["download_url"], stream=True, timeout=request_timeout)
            if response.status_code == 200:
                ftype = response.headers.get('Content-Type')
                if ftype == 'application/zip':
                    archive_file = file + ".zip"
                    with open(archive_file, "wb") as f:
                        f.write(response.raw.read())
                elif ftype == "application/tar.gz":
                    archive_file = file + ".tar.gz"
                    with open(archive_file, "wb") as f:
                        f.write(response.raw.read())

                if ftype == "application/tar.gz":
                    tf = tarfile.open(f"{storage_location}.tar.gz", "r")
                    tf.extractall(f"{CANARY_CORPORA_LOCATION}/{corpus_id}")
                    tf.close()
                elif ftype == "application/zip":
                    with zipfile.ZipFile(f"{storage_location}.zip", "r") as zf:
                        zf.extractall(f"{CANARY_CORPORA_LOCATION}/{corpus_id}")

                logger.info(f"Corpus downloaded to {storage_location}")
                return {
                    "corpus": corpora,
                    "location": storage_location
                }
        except requests.ReadTimeout as e:
            logging.error(
                "The connection timed-out when fetching the corpus. Perhaps try increasing the timeout value.")

    elif corpora_already_downloaded:
        logger.info(f"Corpus already present at {storage_location}")
        return {
            "corpus": corpora,
            "location": storage_location
        }


def load_essay_corpus(purpose=None, merge_claims=False, version=2) -> Union[tuple, list]:
    """Load the essay corpus.

    Parameters
    ----------
    purpose: str
        The purpose for which the corpus is required. Allowed values =
        [
            None,
            'argument_detection',
            'component_prediction',
            "link_prediction",
            'relation_prediction',
            'sequence_labelling'
        ]
    merge_claims: bool
        Whether to merge claims and major claims. Only applies if component_prediction = "component_prediction"
    version: int
        The version of the essay corpus to load

    Returns
    -------
    Union[list, tuple]
        The dataset as either  tuple containing the training labels data or just the parsed essays.
    """

    from ..corpora._essay_corpus import find_paragraph_features, find_cover_sentence_features, find_cover_sentence, \
        tokenize_essay_sentences, find_component_features, relations_in_same_sentence

    _allowed_purpose_values = [
        None,
        'argument_detection',
        'component_prediction',
        "link_prediction",
        'relation_prediction',
        'sequence_labelling'
    ]

    nltk_download(['punkt'])
    _allowed_version_values = [1, 2, "both"]

    if version not in _allowed_version_values:
        raise ValueError(f"{version} is not a valid value. Valid values are {_allowed_version_values}")

    if purpose not in _allowed_purpose_values:
        raise ValueError(f"{purpose} is not a valid value. Valid values are {_allowed_purpose_values}")

    def get_corpus(v: int):
        essay_corpus_location = Path(CANARY_CORPORA_LOCATION) / "argument_annotated_essays_2" if v == 2 else Path(
            CANARY_CORPORA_LOCATION) / "argument_annotated_essays_1"

        if os.path.exists(essay_corpus_location) is False:
            import shutil

            corpus = download_corpus(f"argument_annotated_essays_{v}")
            zip_name = "brat-project-final" if v == 2 else "brat-project"
            corpus_zip = corpus['location'] / f"ArgumentAnnotatedEssays-{v}.0/{zip_name}.zip"
            with zipfile.ZipFile(corpus_zip) as z:
                z.extractall(CANARY_CORPORA_LOCATION)

            # some minor clean up
            os.remove(corpus_zip)
            os.remove(f"{essay_corpus_location}.zip")
            shutil.rmtree(essay_corpus_location)
            os.rename(Path(CANARY_CORPORA_LOCATION) / zip_name, essay_corpus_location)
            if os.path.isdir(f'{CANARY_CORPORA_LOCATION}/__MACOSX'):
                shutil.rmtree(f'{CANARY_CORPORA_LOCATION}/__MACOSX')

        brat_parser = BratParser(error="ignore")
        parsed_essays = brat_parser.parse(essay_corpus_location)

        return parsed_essays

    if version in [1, 2]:
        essays = get_corpus(version)
    else:
        essay_corpus_1 = get_corpus(1)
        essay_corpus_2 = get_corpus(2)
        essays = essay_corpus_1 + essay_corpus_2

    if purpose is None:
        return essays

    elif purpose == "argument_detection":
        data, labels = [], []
        for essay in essays:
            sentences, _labels = [], []
            essay.sentences = tokenize_essay_sentences(essay)
            for sentence in essay.sentences:
                sentences.append(sentence)
                is_argumentative = False
                for component in essay.entities:
                    if component.mention in sentence:
                        is_argumentative = True
                        break
                _labels.append(is_argumentative)
            data += sentences
            labels += _labels

        return data, labels

    elif purpose == "component_prediction":
        data, labels = [], []
        for essay in essays:
            essay.sentences = tokenize_essay_sentences(essay)
            for entity in essay.entities:
                component_feats = {
                    "id": f"{essay.id}_{entity.id}",
                    "essay_ref": essay.id,
                    "ent_ref": entity.id,
                    "component": entity.mention,
                    "cover_sentence": find_cover_sentence(essay, entity),
                }

                cover_sen_comp_split = component_feats['cover_sentence'].split(entity.mention)
                cover_sen_comp_split[0] = nltk.word_tokenize(cover_sen_comp_split[0])
                cover_sen_comp_split[1] = nltk.word_tokenize(cover_sen_comp_split[1])

                component_feats['n_preceding_comp_tokens'] = len(cover_sen_comp_split[0])
                component_feats['n_following_comp_tokens'] = len(cover_sen_comp_split[1])

                component_feats.update({"len_cover_sen": len(nltk.word_tokenize(component_feats['cover_sentence']))})
                component_feats.update(find_component_features(essay, entity))
                data.append(component_feats)
                if merge_claims is False:
                    labels.append(entity.type)
                else:
                    if entity.type == "MajorClaim":
                        data.append("Claim")
                    else:
                        labels.append(entity.type)

        return data, labels

    elif purpose == "link_prediction":

        data, labels = [], []

        for essay in essays:
            _x = []
            _y = []
            _linked = []
            logger.debug(f"Parsing {essay.id}")
            # This could get quite large... depending on n components

            # find paragraph(s) in essay
            paragraphs = [k for k in essay.text.split("\n") if k != ""]
            essay.sentences = tokenize_essay_sentences(essay)

            # loop paragraphs
            for para in paragraphs:
                components = [c for c in essay.entities if c.mention in para]
                relations = [r for r in essay.relations if r.arg2.mention in para and r.arg1.mention in para]

                if len(components) > 0:
                    component_pairs = [tuple(reversed(p)) for p in list(itertools.combinations(components, 2))]
                    for p in component_pairs:
                        arg1, arg2 = p
                        for r in relations:
                            if (arg1.id == r.arg1.id and arg2.id == r.arg2.id) or (
                                    arg2.id == r.arg1.id and arg1.id == r.arg2.id):
                                arg1_feats = find_component_features(essay, arg1, include_link_feats=True)
                                arg2_feats = find_component_features(essay, arg2, include_link_feats=True)

                                feats = {
                                    "source_before_target": arg1_feats['component_position'] > arg2_feats[
                                        'component_position'],
                                    "essay_ref": essay.id,
                                    "para_ref": paragraphs.index(para),
                                    "n_paragraphs": len(paragraphs),
                                    "arg1_in_intro": arg1_feats['is_in_intro'],
                                    "arg1_position": arg1_feats['component_position'],
                                    "arg1_in_conclusion": arg1_feats['is_in_conclusion'],
                                    "arg1_n_preceding_components": arg1_feats['n_preceding_components'],
                                    "arg1_first_in_paragraph": arg1_feats['first_in_paragraph'],
                                    "arg1_last_in_paragraph": arg1_feats['last_in_paragraph'],
                                    "arg1_component": arg1.mention,
                                    "arg1_covering_sentence": find_cover_sentence(essay, arg1),
                                    "arg1_type": arg1.type,
                                    "arg1_n_following_components": arg1_feats['n_following_components'],
                                    "arg2_component": arg2.mention,
                                    "arg2_covering_sentence": find_cover_sentence(essay, arg2),
                                    "arg2_type": arg2.type,
                                    "arg2_position": arg2_feats['component_position'],
                                    "arg2_in_intro": arg2_feats['is_in_intro'],
                                    "arg2_in_conclusion": arg2_feats['is_in_conclusion'],
                                    "arg2_n_following_components": arg2_feats['n_following_components'],
                                    "arg2_n_preceding_components": arg2_feats['n_preceding_components'],
                                    "arg2_first_in_paragraph": arg2_feats['first_in_paragraph'],
                                    "arg2_last_in_paragraph": arg2_feats['last_in_paragraph'],
                                    "arg1_and_arg2_in_same_sentence": relations_in_same_sentence(arg1, arg2, essay),
                                    'arg1_indicator_type_follows_component': arg1_feats[
                                        'indicator_type_follows_component'],
                                    'arg2_indicator_type_follows_component': arg2_feats[
                                        'indicator_type_follows_component'],
                                    'arg1_indicator_type_precedes_component': arg1_feats[
                                        'indicator_type_precedes_component'],
                                    'arg2_indicator_type_precedes_component': arg2_feats[
                                        'indicator_type_precedes_component'],
                                    "n_para_components": len(components),
                                }
                                if feats not in _x:
                                    _linked.append(p)
                                    _x.append(feats)
                                    _y.append("Linked")

            for para in paragraphs:
                components = [c for c in essay.entities if c.mention in para]
                component_pairs = [p for p in list(itertools.permutations(components, 2)) if
                                   p not in _linked]

                for p in component_pairs:
                    arg1, arg2 = p
                    arg1_feats = find_component_features(essay, arg1, include_link_feats=True)
                    arg2_feats = find_component_features(essay, arg2, include_link_feats=True)

                    feats = {
                        "source_before_target": arg1.start > arg2.end,
                        "essay_ref": essay.id,
                        "para_ref": paragraphs.index(para),
                        "n_paragraphs": len(paragraphs),
                        "arg1_in_intro": arg1_feats['is_in_intro'],
                        "arg1_position": arg1_feats['component_position'],
                        "arg1_in_conclusion": arg1_feats['is_in_conclusion'],
                        "arg1_n_preceding_components": arg1_feats['n_preceding_components'],
                        "arg1_first_in_paragraph": arg1_feats['first_in_paragraph'],
                        "arg1_last_in_paragraph": arg1_feats['last_in_paragraph'],
                        "arg1_component": arg1.mention,
                        "arg1_covering_sentence": find_cover_sentence(essay, arg1),
                        "arg1_type": arg1.type,
                        "arg1_n_following_components": arg1_feats['n_following_components'],
                        "arg2_component": arg2.mention,
                        "arg2_covering_sentence": find_cover_sentence(essay, arg2),
                        "arg2_type": arg2.type,
                        "arg2_position": arg2_feats['component_position'],
                        "arg2_in_intro": arg2_feats['is_in_intro'],
                        "arg2_in_conclusion": arg2_feats['is_in_conclusion'],
                        "arg2_n_following_components": arg2_feats['n_following_components'],
                        "arg2_n_preceding_components": arg2_feats['n_preceding_components'],
                        "arg2_first_in_paragraph": arg2_feats['first_in_paragraph'],
                        "arg2_last_in_paragraph": arg2_feats['last_in_paragraph'],
                        "arg1_and_arg2_in_same_sentence": relations_in_same_sentence(arg1, arg2, essay),
                        'arg1_indicator_type_follows_component': arg1_feats[
                            'indicator_type_follows_component'],
                        'arg2_indicator_type_follows_component': arg2_feats[
                            'indicator_type_follows_component'],
                        'arg1_indicator_type_precedes_component': arg1_feats[
                            'indicator_type_precedes_component'],
                        'arg2_indicator_type_precedes_component': arg2_feats[
                            'indicator_type_precedes_component'],
                        "n_para_components": len(components),
                    }

                    if feats not in _x:
                        _x.append(feats)
                        _y.append("Not Linked")

            data += _x
            labels += _y
        return data, labels

    elif purpose == "relation_prediction":
        data = []
        labels = []

        for essay in essays:

            for index, relation in enumerate(essay.relations):
                features = {
                    "essay_id": essay.id,
                    "arg1_component": relation.arg1.mention,
                    "arg2_component": relation.arg2.mention,
                    "arg1_type": relation.arg1.type,
                    "arg2_type": relation.arg2.type,
                    "arg2_position": find_component_features(essay, relation.arg2)['component_position'],
                    "arg1_position": find_component_features(essay, relation.arg1)['component_position'],
                    "n_components_in_essay": len(essay.relations),
                }

                find_cover_sentence_features(features, essay, relation)

                find_paragraph_features(features, relation, essay)

                data.append(features)
                labels.append(relation.type)

        return data, labels

    elif purpose == "sequence_labelling":
        data = []
        labels = []

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
                x1[6][15] = "doing"
                x1[6].insert(16, ".")
                x1[6].insert(17, "In")
                y1[6].append("O")
                y1[6].append("O")

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
                data = x1 + data
                labels = y1 + labels

        # If data and label shapes are not the same, the algorithm will not work.
        # Check this ahead of time
        errors = 0
        for j in zip(data, labels):
            if len(j[0]) != len(j[1]):
                errors += 1

        if errors > 0:
            raise ValueError(f'Data is incorrect shape. Number of errors {errors}')

        return data, labels


def load_araucaria_corpus() -> dict:
    """Loads the araucaria corpus from aifdb

    Returns
    -------
    dict:
        The araucaria corpus
    """

    corpus_download = download_corpus("araucaria", aifdb_corpus=True)
    if "location" in corpus_download:

        corpus_location = corpus_download["location"]
        files = glob.glob(str(Path(corpus_location) / "nodeset*.json"))

        for i, file in enumerate(files):
            file = Path(file)
            files[i] = {'json': None, 'text': None}

            with open(file, "r", encoding="utf8") as json_file:
                files[i]['json'] = json.load(json_file)

            with open(str(Path(corpus_location / f"{file.stem}.txt")), "r", encoding="utf8") as text_file:
                files[i]['text'] = text_file.read()

        return files
