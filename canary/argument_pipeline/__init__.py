import os
import zipfile
from pathlib import Path

import joblib
import requests

import canary
from canary import logger
from canary.utils import CANARY_MODEL_DOWNLOAD_LOCATION, CANARY_MODEL_STORAGE_LOCATION


def download_pretrained_models(location=None, download_to=None, overwrite=False):
    """
    Download the pretrained models from a web location

    :param location: the url of the model zip archive
    :param download_to: an alternative place to download the models
    :param overwrite: overwrite existing models
    """

    # Make sure the canary local directory(s) exist beforehand
    os.makedirs(CANARY_MODEL_STORAGE_LOCATION, exist_ok=True)

    def unzip_models():
        with zipfile.ZipFile(download_to, "r") as zf:
            zf.extractall(Path(CANARY_MODEL_STORAGE_LOCATION))
            logger.debug("models extracted.")

    if location is None:
        location = CANARY_MODEL_DOWNLOAD_LOCATION

    if download_to is None:
        download_to = Path(CANARY_MODEL_STORAGE_LOCATION / "models.zip")

    model_zip_present = os.path.isfile(download_to)

    # Let's get the pretrained stuff
    if model_zip_present is False or overwrite is True:
        response = requests.get(location, stream=True)

        if response.status_code == 200:
            with open(download_to, "wb") as f:
                f.write(response.raw.read())
                f.close()
                logger.debug("Models downloaded! Attempting to unzip.")

            unzip_models()
        else:
            logger.error(f"There was an issue getting the models. Http response code: {response.status_code}")
    elif model_zip_present is True:
        logger.debug("model zip already present.")
        unzip_models()


def analyse(doc: str = None, docs: list = None, out: str = "str", **kwargs):
    """

    :param docs:
    :param doc:
    :param kwargs:
    :return:
    """

    allowed_out_values = ["str"]
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
