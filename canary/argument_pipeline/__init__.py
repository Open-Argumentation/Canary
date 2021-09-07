import glob
import itertools
import os
import zipfile
from pathlib import Path

import joblib
import requests

import canary
import canary.corpora
from canary.utils import CANARY_MODEL_DOWNLOAD_LOCATION, CANARY_MODEL_STORAGE_LOCATION


def get_downloadable_assets_from_github():
    canary_repo = canary.utils.config.get('canary', 'model_download_location')
    res = requests.get(f"https://api.github.com/repos/{canary_repo}/releases/tags/latest",
                       headers={"Accept": "application/vnd.github.v3+json"})
    if res.status_code == 200:
        import json
        return json.loads(res.text)
    else:
        canary.utils.logger.error("Could not get details from github")
        return None


def get_models_not_on_disk():
    current_models = models_available_on_disk()
    github_assets = [j['name'].split(".")[0] for j in get_downloadable_assets_from_github()['assets']]
    return list(set(github_assets) - set(current_models))


def models_available_on_disk() -> list:
    from glob import glob
    return [Path(s).stem for s in glob(str(CANARY_MODEL_STORAGE_LOCATION / "*.joblib"))]


def download_pretrained_models(model: str, location=None, download_to=None, overwrite=False):
    """
    Download the pretrained models from a GitHub.

    :param model. The model(s) to download. Defaults to "all"
    :param location: the url of the model zip archive. Defaults to None
    :param download_to: an alternative place to download the models. Defaults to None
    :param overwrite: overwrite existing models. Defaults to False.
    """

    # Make sure the canary local directory(s) exist beforehand
    os.makedirs(CANARY_MODEL_STORAGE_LOCATION, exist_ok=True)

    if location is None:
        location = CANARY_MODEL_DOWNLOAD_LOCATION

    if download_to is None:
        download_to = Path(CANARY_MODEL_STORAGE_LOCATION)

    if type(model) not in [str, list]:
        raise ValueError("Model value should be a string or a list")

    def unzip_model(model_zip: str):
        with zipfile.ZipFile(model_zip, "r") as zf:
            zf.extractall(download_to)
            canary.utils.logger.info("models extracted.")
        os.remove(model_zip)

    def download_asset(asset):
        if 'url' in asset:
            asset_res = requests.get(asset['url'], headers={
                "Accept": "application/octet-stream"
            }, stream=True)

            if asset_res.status_code == 200:
                file = download_to / asset['name']
                with open(file, "wb") as f:
                    f.write(asset_res.raw.read())
                    canary.utils.logger.info("downloaded model")

                if file.suffix == ".zip":
                    unzip_model(download_to / asset['name'])

                if file.suffix == ".joblib":
                    canary.utils.logger.info(f"{file.name} downloaded to {file.parent}")
            else:
                canary.utils.logger.error("There was an error downloading the asset.")
                return

    # check if we have already have the model downloaded
    if overwrite is False:
        models = glob.glob(str(download_to / "*.joblib"))
        if len(models) > 0:
            for model in models:
                if os.path.isfile(model) is True:
                    canary.utils.logger.info("This model already exists")
                    print("This model already exists")
                    return

    github_releases = get_downloadable_assets_from_github()
    if github_releases is not None:
        # parse JSON response
        if 'assets' in github_releases:
            if len(github_releases['assets']) > 0:
                for asset in github_releases['assets']:
                    name = asset['name'].split(".")[0]
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


def analyse_file(file, out_format: str = "stdout", steps=None):
    supported_file_types = ["txt"]

    if not os.path.isfile(file):
        raise TypeError("file argument should be a valid file")

    # file_path = Path(file)
    # if file_path.suffix

    with open(file, "r", encoding='utf-8') as document:
        return analyse(document.read(), out_format=out_format, steps=steps)


def analyse(document: str, out_format=None, steps=None, **kwargs):
    """
    """

    allowed_output_formats = [None, "stdout", "json", "csv"]

    from canary.argument_pipeline.argument_segmenter import ArgumentSegmenter
    from canary.argument_pipeline.component_identification import ArgumentComponent

    if steps is None:
        canary.utils.logger.debug("Default pipeline being used")
        steps = [""]

    canary.utils.logger.debug(document)
    allowed_out_format_values = ['stdout', 'json', 'csv']
    if type(out_format) is not str:
        raise TypeError("Parameter out_format should be a string")

    if out_format not in allowed_out_format_values:
        raise ValueError(f"Incorrect out_format value. Allowed values are {allowed_out_format_values}")

    if type(document) is not str:
        raise TypeError("The inputted document should be a string.")

    # load segmenter
    canary.utils.logger.debug("Loading Argument Segmenter")
    segmenter: ArgumentSegmenter = load('arg_segmenter')

    if segmenter is None:
        canary.utils.logger.error("Failed to load segmenter")
        raise ValueError("Could not load segmenter.")

    components = segmenter.get_components_from_document(document)

    if len(components) > 0:
        canary.utils.logger.debug("Loading component predictor.")
        component_predictor: ArgumentComponent = canary.load('argument_component')

        if component_predictor is None:
            raise TypeError("Could not load argument component predictor")

        for component in components:
            component['type'] = component_predictor.predict(component)

        from canary.argument_pipeline.link_predictor import LinkPredictor
        link_predictor: LinkPredictor = canary.load('link_predictor')

        if link_predictor is None:
            raise TypeError("Could not load link predictor")

        all_possible_component_pairs = [tuple(reversed(j)) for j in list(itertools.permutations(components, 2)) if
                                        j[0] != j[1]]
        canary.utils.logger.debug(f"{len(all_possible_component_pairs)} possible combinations.")

        sentences = canary.corpora.essay_corpus.tokenize_essay_sentences(document)
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
            }
            args_linked = link_predictor.predict(all_possible_component_pairs[i]) == "Linked"
            all_possible_component_pairs[i].update({"args_linked": args_linked})

        canary.utils.logger.debug("Done")
        linked_relations = [pair for pair in all_possible_component_pairs if pair["args_linked"] is True]
        canary.utils.logger.debug(
            f" {len(linked_relations)} / {len(all_possible_component_pairs)} identified as being linked")

        # Find attack / support relations
        if len(linked_relations) > 0:
            from canary.argument_pipeline.structure_prediction import StructurePredictor
            sp: StructurePredictor = canary.load('structure_predictor')
            for r in linked_relations:
                r['scheme'] = sp.predict(r)

        support_relations = [pair for pair in linked_relations if pair["scheme"] == 'supports']
        attacks_relations = [pair for pair in linked_relations if pair["scheme"] == 'attacks']

        if out_format == "json":
            import json
            return json.dumps(components)

        return components, linked_relations, support_relations, attacks_relations

    else:
        canary.utils.logger.warn("Didn't find any evidence of argumentation")


def load(model_id: str, model_dir=None, download_if_missing=False, **kwargs):
    """
    load a model

    :param download_if_missing:
    :param model_dir:
    :param model_id:
    :return:
    """

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
            f"Did not manage to load the model specified. Available models on disk are: {models_available_on_disk()}.")
        models_not_on_disk = get_models_not_on_disk()
        if len(models_not_on_disk) > 0:
            canary.utils.logger.error(f"Models available via download are: {models_not_on_disk}.")
