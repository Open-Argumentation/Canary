import glob
import os
import zipfile
from functools import cache
from pathlib import Path

import joblib
import requests

import canary
from canary.utils import CANARY_MODEL_DOWNLOAD_LOCATION, CANARY_MODEL_STORAGE_LOCATION


@cache
def get_downloadable_assets_from_github():
    canary_repo = canary.config.get('canary', 'model_download_location')
    res = requests.get(f"https://api.github.com/repos/{canary_repo}/releases/tags/latest",
                       headers={"Accept": "application/vnd.github.v3+json"})
    if res.status_code == 200:
        import json
        return json.loads(res.text)
    else:
        canary.logger.error("Could not get details from github")
        return None


def get_models_not_on_disk():
    current_models = models_available_on_disk()
    github_assets = [j['name'].split(".")[0] for j in get_downloadable_assets_from_github()['assets']]
    return list(set(github_assets) - set(current_models))


def models_available_on_disk() -> list:
    from glob import glob
    return [Path(s).stem for s in glob(str(CANARY_MODEL_STORAGE_LOCATION / "*.joblib"))]


@cache
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
            canary.logger.info("models extracted.")
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
                    canary.logger.info("downloaded model")

                if file.suffix == ".zip":
                    unzip_model(download_to / asset['name'])

                if file.suffix == ".joblib":
                    canary.logger.info(f"{file.name} downloaded to {file.parent}")
            else:
                canary.logger.error("There was an error downloading the asset.")
                return

    # check if we have already have the model downloaded
    if overwrite is False:
        models = glob.glob(str(download_to / "*.joblib"))
        if len(models) > 0:
            for model in models:
                if os.path.isfile(model) is True:
                    canary.logger.info("This model already exists")
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
                canary.logger.info("No assets to download.")
                return
        if 'prerelease' in github_releases:
            canary.logger.info("This has been marked as a pre-release.")
    else:
        canary.logger.error(f"There was an issue getting the models")


@cache
def analyse(document: str, out_format: str = "stdout", steps=None, **kwargs):
    """
    """
    from canary.argument_pipeline.argument_segmenter import ArgumentSegmenter
    from canary.argument_pipeline.component_identification import ArgumentComponent

    if steps is None:
        canary.logger.debug("Default pipeline being used")
        steps = [""]

    canary.logger.debug(document)
    allowed_out_format_values = ['stdout', 'json', 'csv']
    if type(out_format) is not str:
        raise TypeError("Parameter out_format should be a string")

    if out_format not in allowed_out_format_values:
        raise ValueError(f"Incorrect out_format value. Allowed values are {allowed_out_format_values}")

    if type(document) is not str:
        raise TypeError("The inputted document should be a string.")

    # load segmenter
    canary.logger.debug("Loading Argument Segmenter")
    segmenter: ArgumentSegmenter = load('arg_segmenter')

    if segmenter is None:
        canary.logger.error("Failed to load segmenter")
        raise ValueError("Could not load segmenter.")

    components = segmenter.get_components_from_document(document)

    if len(components) > 0:
        canary.logger.debug("Loading component predictor.")
        component_predictor: ArgumentComponent = canary.load('argument_component')
        if component_predictor is None:
            raise TypeError("Could not load argument component predictor")

        for component in components:
            component['type'] = component_predictor.predict(component)

        canary.logger.debug("Done")
        return components
    else:
        canary.logger.warn("Didn't find any evidence of argumentation")
        return


@cache
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
            canary.logger.warn("There does not appear to be a model here. This may fail.")
    else:
        absolute_model_path = Path(CANARY_MODEL_STORAGE_LOCATION) / model_id

    if os.path.isfile(absolute_model_path):
        canary.logger.debug(f"Loading {model_id} from {absolute_model_path}")
        return joblib.load(absolute_model_path)
    else:
        canary.logger.error(
            f"Did not manage to load the model specified. Available models on disk are: {models_available_on_disk()}")
