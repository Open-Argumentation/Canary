import glob
import os
import zipfile
from pathlib import Path

import joblib
import requests

import canary
from canary.utils import CANARY_MODEL_DOWNLOAD_LOCATION, CANARY_MODEL_STORAGE_LOCATION


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

    # Use GitHub's REST API to find releases.
    import json
    res = requests.get(f"https://api.github.com/repos/{location}/releases/tags/latest",
                       headers={"Accept": "application/vnd.github.v3+json"})
    if res.status_code == 200:
        # parse JSON response
        payload = json.loads(res.text)
        if 'assets' in payload:
            if len(payload['assets']) > 0:
                for asset in payload['assets']:
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
        if 'prerelease' in payload:
            canary.logger.info("This has been marked as a pre-release.")
    else:
        canary.logger.error(f"There was an issue getting the models. Http response code: {res.status_code}")


def analyse(doc: str = None, docs: list = None, out_format: str = "stdout", **kwargs):
    """

    :param out_format:
    :param docs:
    :param doc:
    :param kwargs:
    :return:
    """

    if type(out_format) is not str:
        pass

    if out_format not in ['stdout', 'json', 'csv']:
        pass

    if doc is not None and docs is not None:
        raise ValueError("come one man")

    if doc is not None and type(doc) is not str:
        raise ValueError("")

    if docs is not None and type(docs) is not list:
        raise ValueError("")

    from canary.argument_pipeline.argument_segmenter import ArgumentSegmenter
    import nltk

    if doc is not None:
        argumentative_sentences = []
        non_argumentative_sentences = []

        sentences = nltk.sent_tokenize(doc)
        for s in sentences:
            if ArgumentSegmenter().predict(s, binary=True) is False:
                non_argumentative_sentences.append(s)
            else:
                argumentative_sentences.append(s)


def load(model_id: str, **kwargs):
    """
    load a model

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
        return joblib.load(absolute_model_path)
    else:
        canary.logger.error("Did not manage to load the model specified.")
