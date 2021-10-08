""" The argument_pipeline package contains the functionality required to analyse a unstructured document
of natural language.
"""
import glob
import itertools
import os
import zipfile
from pathlib import Path

import joblib
import requests

__all__ = [
    "load",
    "analyse",
    "analyse_file",
    "download"
]


def get_downloadable_assets_from_github():
    """Finds downloadable argument models which are available from Canary's GitHub repository

    Returns
    -------
    str
        The JSON string from the GitHub REST API
    """
    from canary.utils import CANARY_MODEL_DOWNLOAD_LOCATION
    import canary
    canary_repo = canary.utils.config.get('canary', 'model_download_location')
    res = requests.get(f"https://api.github.com/repos/{canary_repo}/releases/tags/latest",
                       headers={"Accept": "application/vnd.github.v3+json"})
    if res.status_code == 200:
        import json
        return json.loads(res.text)
    else:
        canary.utils.logger.error("Could not get details from github")
        return None


def _get_models_not_on_disk():
    current_models = _models_available_on_disk()
    github_assets = [j['name'].split(".")[0] for j in get_downloadable_assets_from_github()['assets']]
    return list(set(github_assets) - set(current_models))


def _models_available_on_disk() -> list:
    from canary.utils import CANARY_MODEL_STORAGE_LOCATION

    from glob import glob
    return [Path(s).stem for s in glob(str(CANARY_MODEL_STORAGE_LOCATION / "*.joblib"))]


def download(model: str, download_to: str = None, overwrite=False):
    """Downloads the pretrained Canary models from a GitHub.

    Parameters
    ----------
    model: str
        The model ID to download.
    download_to: str
        Where to download the model to.
    overwrite: bool, default=False
        Should Canary overwrite existing models if they are already present.
    """
    import canary
    from canary.utils import CANARY_MODEL_DOWNLOAD_LOCATION, CANARY_MODEL_STORAGE_LOCATION

    # Make sure the canary local directory(s) exist beforehand
    os.makedirs(CANARY_MODEL_STORAGE_LOCATION, exist_ok=True)

    if download_to is None:
        download_to = Path(CANARY_MODEL_STORAGE_LOCATION)

    if type(model) not in [str, list]:
        raise ValueError("Model value should be a string or a list")

    def unzip_model(model_zip: str):
        with zipfile.ZipFile(model_zip, "r") as zf:
            zf.extractall(download_to)
            canary.utils.logger.info("models extracted.")
        os.remove(model_zip)

    def download_asset(asset_dict):
        if 'url' in asset_dict:
            asset_res = requests.get(asset_dict['url'], headers={
                "Accept": "application/octet-stream"
            }, stream=True)

            if asset_res.status_code == 200:
                file = download_to / asset_dict['name']
                with open(file, "wb") as f:
                    f.write(asset_res.raw.read())
                    canary.utils.logger.info("downloaded model")

                if file.suffix == ".zip":
                    unzip_model(download_to / asset_dict['name'])

                if file.suffix == ".joblib":
                    canary.utils.logger.info(f"{file.name} downloaded to {file.parent}")
            else:
                canary.utils.logger.error("There was an error downloading the asset.")
                return

    # check if we have already have the model downloaded
    if overwrite is False and model != "all":
        models = glob.glob(str(download_to / "*.joblib"))
        if len(models) > 0:
            for model in models:
                if os.path.isfile(model) is True:
                    canary.utils.logger.warn(f"{Path(model).stem} already present: {model}")
                    return

    github_releases = get_downloadable_assets_from_github()
    models_on_disk = _models_available_on_disk()
    if github_releases is not None:
        # parse JSON response
        if 'assets' in github_releases:
            if len(github_releases['assets']) > 0:
                for asset in github_releases['assets']:
                    name = asset['name'].split(".")[0]
                    if name in models_on_disk and overwrite is False:
                        canary.utils.logger.warn(f"{name} already present.")
                        continue
                    if type(model) is str:
                        if model != "all":
                            if name == model:
                                download_asset(asset)
                        elif model == "all":
                            download_asset(asset)
                    if type(model) is list:
                        if all(type(j) is str for j in model) is True:
                            if name in model:
                                download_asset(asset)
                        else:
                            raise ValueError("All items in the list should be strings.")
            else:
                canary.utils.logger.info("No assets to download.")
                return
        if 'prerelease' in github_releases:
            canary.utils.logger.info("This has been marked as a pre-release.")
    else:
        canary.utils.logger.error(f"There was an issue getting the models")


def analyse_file(file, min_link_confidence=0.8, min_support_confidence=0.8, min_attack_confidence=0.8,
                 **kwargs):
    """Wrapper around the `analyse` function which takes in a file location as a string.

    Parameters
    ----------
    file: str
        The absolute file path
    min_link_confidence: float, default=0.8
        The minimum confidence needed for two arguments to be considered "linked"
    min_support_confidence: float, default=0.8
        The minimum confidence needed to be classified as a support relation
    min_attack_confidence: float, default=0.8
        The minimum confidence needed to be classified as an attack relation
    **kwargs: dict
        Keyword arguments

    Returns
    -------
    dict
        the SADFace document.

    Examples
    --------
    >>> from canary import analyse_file
    >>> document = "/Users/my_user/doc.txt"
    >>> analysis = analyse_file(document)
    >>> analysis
    {
        "metadata": {...},
        "resources": {...},
        "nodes": {...},
        "edges": {...}
    }

    Notes
    -----

    Refer to https://github.com/ARG-ENU/SADFace
    """

    if not os.path.isfile(file):
        raise TypeError("file argument should be a valid file")

    with open(file, "r", encoding='utf-8') as document:
        return analyse(
            document.read(),
            min_link_confidence=min_link_confidence,
            min_support_confidence=min_support_confidence,
            min_attack_confidence=min_attack_confidence,
        )


def analyse(document: str, min_link_confidence=0.8, min_support_confidence=0.8,
            min_attack_confidence=0.8, **kwargs):
    r"""Analyses a document .

    Parameters
    ----------
    document: str
        The document text that is being analysed
    min_link_confidence: float, default=0.8
        The minimum confidence needed for two arguments to be considered "linked"
    min_support_confidence: float, default=0.8
        The minimum confidence needed to be classified as a support relation
    min_attack_confidence: float, default=0.8
        The minimum confidence needed to be classified as an attack relation
    **kwargs: dict
        Keyword arguments

    Returns
    -------
    dict
        the SADFace document.

    Examples
    --------
    >>> from canary import analyse
    >>> document_text = "..."
    >>> analysis = analyse(document_text, min_link_confidence=0.65)
    >>> analysis
    {
        "metadata": {...},
        "resources": {...},
        "nodes": {...},
        "edges": {...}
    }

    Notes
    -----

    Refer to https://github.com/ARG-ENU/SADFace
    """
    from ..argument_pipeline.argument_segmentation import ArgumentSegmenter
    from ..argument_pipeline.component_prediction import ArgumentComponent
    from ..utils import logger
    from .. import __version__
    from ..corpora import _essay_corpus

    logger.debug(document)

    if type(document) is not str:
        raise TypeError("The inputted document should be a string.")

    # load segmenter
    logger.debug("Loading Argument Segmenter")
    segmenter: ArgumentSegmenter = load('arg_segmenter')

    if segmenter is None:
        logger.error("Failed to load segmenter")
        raise ValueError("Could not load segmenter.")

    components = segmenter.get_components_from_document(document)

    if len(components) > 0:
        n_claims, n_major_claims, n_premises = 0, 0, 0
        logger.debug("Loading component predictor.")
        component_predictor: ArgumentComponent = load('argument_component')

        if component_predictor is None:
            raise TypeError("Could not load argument component predictor")

        for component in components:
            component['type'] = component_predictor.predict(component)
            if component['type'] == 'Claim':
                n_claims += 1
            elif component['type'] == "MajorClaim":
                n_major_claims += 1
            elif component['type'] == "Premise":
                n_premises += 1

        from canary.argument_pipeline.link_predictor import LinkPredictor
        link_predictor: LinkPredictor = load('link_predictor')

        if link_predictor is None:
            raise TypeError("Could not load link predictor")

        all_possible_component_pairs = [tuple(reversed(j)) for j in list(itertools.permutations(components, 2)) if
                                        j[0] != j[1]]
        logger.debug(f"{len(all_possible_component_pairs)} possible combinations.")

        sentences = _essay_corpus.tokenize_essay_sentences(document)
        for i, pair in enumerate(all_possible_component_pairs):
            arg1, arg2 = pair
            same_sentence = False

            for s in sentences:
                if arg1['component'] in s and arg2['component'] in s:
                    same_sentence = True

            all_possible_component_pairs[i] = {
                "source_before_target": arg1['component_position'] > arg2['component_position'],
                "arg1_component": arg1["component"],
                "arg2_component": arg2["component"],
                "arg1_covering_sentence": arg1["cover_sentence"],
                "arg2_covering_sentence": arg2["cover_sentence"],
                "arg1_n_preceding_components": arg1['n_preceding_components'],
                "arg1_n_following_components": arg1['n_following_comp_tokens'],
                "arg2_n_preceding_components": arg2['n_preceding_components'],
                "arg2_n_following_components": arg2['n_following_comp_tokens'],
                "arg1_first_in_paragraph": arg1["first_in_paragraph"],
                "arg2_first_in_paragraph": arg2["first_in_paragraph"],
                "arg2_last_in_paragraph": arg2['last_in_paragraph'],
                "arg1_last_in_paragraph": arg1['last_in_paragraph'],
                "arg1_in_intro": arg1["is_in_intro"],
                "arg2_in_intro": arg2["is_in_intro"],
                "arg1_in_conclusion": arg1["is_in_conclusion"],
                "arg2_in_conclusion": arg2["is_in_conclusion"],
                "arg1_type": arg1["type"],
                "arg2_type": arg2["type"],
                "arg1_and_arg2_in_same_sentence": same_sentence,
                "n_para_components": arg1['n_following_components'] + arg2['n_preceding_components'],
                "arg1_position": arg1['component_position'],
                "arg2_position": arg2['component_position'],
                'arg1_indicator_type_follows_component': arg1['indicator_type_follows_component'],
                'arg2_indicator_type_follows_component': arg2['indicator_type_follows_component'],
                'arg1_indicator_type_precedes_component': arg1['indicator_type_precedes_component'],
                'arg2_indicator_type_precedes_component': arg2['indicator_type_precedes_component']

            }
            args_linked = link_predictor.predict(all_possible_component_pairs[i]) == "Linked"
            all_possible_component_pairs[i].update({"args_linked": args_linked})

        logger.debug("Done")
        linked_relations = [pair for pair in all_possible_component_pairs if
                            pair["args_linked"] is True and link_predictor.predict(pair, probability=True)[
                                "Linked"] >= min_link_confidence]

        logger.debug(
            f" {len(linked_relations)} / {len(all_possible_component_pairs)} identified as being linked")

        # Find attack / support relations
        if len(linked_relations) > 0:
            from canary.argument_pipeline.structure_prediction import StructurePredictor
            sp: StructurePredictor = load('structure_predictor')
            for r in linked_relations:
                r['scheme'] = sp.predict(r)

        support_relations = [pair for pair in linked_relations if pair["scheme"] == 'supports']
        attacks_relations = [pair for pair in linked_relations if pair["scheme"] == 'attacks']

        logger.debug(f"Number of attack relations: {len(attacks_relations)}")
        logger.debug(f"Number of support relations: {len(support_relations)}")

        # Create a sadface document for what we have found
        from sadface import sadface
        sadface.initialise()
        sadface.set_title("doc")

        # Create atoms
        for c in components:
            atom = sadface.add_atom(c['component'])
            sadface.add_atom_metadata(atom['id'], 'canary', 'type', c['type'])

        # Add edges
        for l in attacks_relations:
            arg1_id = sadface.get_atom_id(l['arg1_component'])
            arg2_id = sadface.get_atom_id(l['arg2_component'])
            sadface.add_conflict(arg_id=arg1_id, conflict_id=arg2_id)

        for l in support_relations:
            arg1_id = sadface.get_atom_id(l['arg1_component'])
            arg2_id = sadface.get_atom_id(l['arg2_component'])
            sadface.add_support(con_id=arg1_id, prem_id=[arg2_id])

        # add some nice metadata
        sadface.add_global_metadata('canary', 'number_of_components', len(components))
        sadface.add_global_metadata('canary', 'number_of_attack_relations', len(attacks_relations))
        sadface.add_global_metadata('canary', 'number_of_support_relations', len(support_relations))
        sadface.add_global_metadata('canary', 'number_of_linked_relations', len(linked_relations))
        sadface.add_global_metadata('canary', 'number_of_premises', n_premises)
        sadface.add_global_metadata('canary', 'number_of_claims', n_claims)
        sadface.add_global_metadata('canary', 'number_of_major_claims', n_major_claims)
        sadface.add_global_metadata('canary', 'version', __version__)

        return sadface.get_document()

    else:
        import canary.utils
        canary.utils.logger.warn("Didn't find any evidence of argumentation")


def load(model_id: str, model_dir=None, download_if_missing=False, **kwargs):
    """Load a trained Canary model from disk.

    Parameters
    ----------
    model_id: str
        The ID of the model to download
    model_dir: str
        Where the model should be loaded from
    download_if_missing: bool
        Should Canary attempt to download the model if it is not present on disk?
    kwargs: dict
        Keyword arguments

    Returns
    -------
    Model
        The Canary model.

    Examples
    --------
    >>> import canary
    >>> component_detector = canary.load("argument_component")
    >>> print(component_detector.__class__.__name__)
    """
    import canary
    from canary.utils import CANARY_MODEL_DOWNLOAD_LOCATION, CANARY_MODEL_STORAGE_LOCATION

    if ".joblib" not in model_id:
        model_id = model_id + ".joblib"

    if 'model_dir' in kwargs:
        absolute_model_path = Path(kwargs["model_dir"]) / model_id
        if os.path.isfile(absolute_model_path) is False:
            canary.utils.logger.warn("There does not appear to be a model here. This may fail.")
    else:
        absolute_model_path = Path(CANARY_MODEL_STORAGE_LOCATION) / model_id

    if os.path.isfile(absolute_model_path):
        canary.utils.logger.debug(f"Loading {model_id} from {absolute_model_path}")
        return joblib.load(absolute_model_path)
    else:
        canary.utils.logger.error(
            f"Did not manage to load the model specified. Available models on disk are: {_models_available_on_disk()}.")
        models_not_on_disk = _get_models_not_on_disk()
        if len(models_not_on_disk) > 0:
            canary.utils.logger.error(f"Models available via download are: {models_not_on_disk}.")
